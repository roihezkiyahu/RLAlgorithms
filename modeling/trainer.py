import torch
import torch.nn as nn
import torch.optim as optim
import os
from modeling.memory import ReplayMemory, Transition, PERMemory
import math
import numpy as np
from modeling.processing import preprocess_state, postprocess_action
import torch.nn.functional as F
import random
import imageio
import matplotlib
try:
    matplotlib.use('Agg')
except:
    print("no TkAgg")
import matplotlib.pyplot as plt
import csv
from modeling.AtariGameWrapper import AtariGameWrapper, AtariGameViz
from torch.nn import utils
import yaml
import threading
import queue
from modeling.logger import Logger


class Debugger:
    def __init__(self, agent):
        self.agent = agent
        self.loss_history = np.array([])
        self.gradient_norms = np.array([])
        self.score_history = np.array([])
        self.epsilons = np.array([])

    def track_loss(self, loss):
        self.loss_history = np.append(self.loss_history, loss.item())

    def track_gradients(self):
        gradients = []
        for p in self.agent.model.parameters():
            if p.grad is not None:
                gradients.append(p.grad.norm().item())
        mean_gradient = np.mean(gradients)
        self.gradient_norms = np.append(self.gradient_norms, mean_gradient)

    def track_scores(self, score):
        self.score_history = np.append(self.score_history, score)

    def track_epsilon(self, epsilon):
        self.epsilons = np.append(self.epsilons, epsilon)

    @staticmethod
    def moving_average(a, window=3):
        ret = np.cumsum(a, dtype=float)
        ret[window:] = ret[window:] - ret[:-window]
        return ret[window - 1:] / window

    def plot_loss(self, episodes, subplot=221, window=500):
        plt.subplot(subplot)
        plt.plot(episodes, self.loss_history, label='Loss')
        running_avg = Debugger.moving_average(self.loss_history, window)
        n_epochs = len(episodes)
        plt.plot(range(window, n_epochs + 1), running_avg, label=f'Loss Running Average {window}',
                 linestyle='dashed')
        plt.title('Losses over Time')
        plt.xlabel('Optimization step')
        plt.ylabel('Loss')
        plt.legend()

    def plot_grads(self, episodes, subplot=222, window=500):
        plt.subplot(subplot)
        plt.plot(episodes, self.gradient_norms, label='Gradient Norms')
        running_avg = Debugger.moving_average(self.gradient_norms, window)
        n_epochs = len(episodes)
        plt.plot(range(window, n_epochs + 1), running_avg, label=f'Norms Running Average {window}',
                 linestyle='dashed')
        plt.title('Gradient Norms over Time')
        plt.xlabel('Optimization step')
        plt.ylabel('Gradient Norm')
        plt.legend()

    def plot_epsilon(self, subplot=223):
        plt.subplot(subplot)
        plt.plot(range(1, len(self.epsilons) + 1), self.epsilons, label='Epsilon')
        plt.title('Epsilon over Time')
        plt.xlabel('Optimization step')
        plt.ylabel('Epsilon')
        plt.legend()

    def plot_scores(self, window, subplot=224): #212
        plt.subplot(subplot)
        scores = self.score_history
        n_scores = len(scores)
        window = min(window, n_scores)
        running_avg = self.moving_average(scores, window)
        running_avg_3 = self.moving_average(scores, min(window * 3, n_scores))
        running_avg_5 = self.moving_average(scores, min(window * 5, n_scores))
        plt.plot(range(1, n_scores + 1), scores, label='Score')
        plt.plot(range(window, n_scores + 1), running_avg, label=f'Running Average {window}', linestyle='dashed')
        plt.plot(range(window * 3, n_scores + 1), running_avg_3, label=f'Running Average {window * 3}', linestyle='dashed')
        plt.plot(range(window * 5, n_scores + 1), running_avg_5, label=f'Running Average {window * 5}', linestyle='dashed')
        plt.title('Rewards History')
        plt.xlabel('Game Number')
        plt.ylabel('Score')
        plt.legend()

    def plot_diagnostics(self, epoch, window=100):
        episodes = range(len(self.loss_history))
        plt.figure(figsize=(15, 10))
        self.plot_loss(episodes)
        self.plot_grads(episodes)
        self.plot_epsilon()
        self.plot_scores(window)
        plt.tight_layout()
        filename = f"{self.agent.prefix_name}_{epoch}_diagnostics.png"
        plt.savefig(filename)
        print(f"Saved diagnostics to {filename}")
        plt.close()


class Trainer:
    def __init__(self, config_path, model, clone_model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(config_path, str):
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)['trainer']
        else:
            config = config_path
        self.episodes = int(config['episodes'])
        self.learning_rate = float(config['learning_rate'])
        self.gamma = float(config['gamma'])
        self.n_memory_episodes = int(config['n_memory_episodes'])
        self.save_gif_every_x_epochs = int(config['save_gif_every_x_epochs'])
        self.batch_size = int(config['batch_size'])
        self.EPS_START = float(config['EPS_START'])
        self.EPS_END = float(config['EPS_END'])
        self.EPS_DECAY = float(config['EPS_DECAY'])
        self.TAU = float(config['TAU'])
        self.max_episode_len = int(config['max_episode_len'])
        self.use_ddqn = bool(config['use_ddqn'])
        self.validate_every_n_episodes = int(config['validate_every_n_episodes'])
        self.validate_episodes = int(config['validate_episodes'])
        self.n_actions = int(config['n_actions'])
        self.game_wrapper = config['game_wrapper']
        self.game = self.game_wrapper.game
        self.visualizer = config['visualizer']
        self.fps = int(config['gif_fps'])
        self.reset_options = config['reset_options']
        self.update_every_n_steps = int(config['update_every_n_steps'])
        self.update_target_every_n_steps = int(config['update_target_every_n_steps'])
        self.save_diagnostics = int(config['save_diagnostics'])
        self.clip_grad = float(config['clip_grad'])
        self.save_model_every_n = int(config['save_model_every_n'])
        self.warmup_steps = int(config['warmup_steps'])
        self.probs = config.get('prior_probs', None)
        if isinstance(self.game_wrapper, type(None)):
            self.game_wrapper = AtariGameWrapper(self.game)
        if isinstance(self.visualizer, type(None)):
            self.visualizer = AtariGameViz(self.game, self.device)
        self.model = model.to(self.device)
        self.target_net = clone_model.to(self.device)
        if config['folder']:
            os.makedirs(config['folder'], exist_ok=True)
            self.prefix_name = os.path.join(config['folder'], config['prefix_name'])
        else:
            self.prefix_name = config['prefix_name']

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate) if not config['optimizer'] else config['optimizer']
        self.use_scheduler = bool(config['use_scheduler'])
        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5,
                                                                        patience=int(config['patience']),
                                                                        threshold=0.1, verbose=True)
        self.criterion = nn.MSELoss()
        self.rewards_memory = []
        self.score_memory = []
        self.frames = []
        self.memory = PERMemory(int(config['replaymemory']), float(config['per_alpha']))
        self.validation_log = []
        self.debugger = Debugger(self)
        self.best_model = {"model": None, "score": - np.inf}
        self.total_steps = 0
        if config.get('log_results', None) is not None:
            self.logger = Logger(apply=True, verbose=False, output_folder=self.prefix_name)
        else:
            self.logger = Logger(apply=False, verbose=False, output_folder=self.prefix_name)
        self.early_stopping = config.get('early_stopping', 0)

    def choose_action(self, state, epsilon, validation=False):
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:
            state_tensor = state.unsqueeze(0).to(self.device)

        if validation:
            with torch.no_grad():
                probs = self.model(state_tensor)
                return torch.argmax(probs).item(), torch.round(F.softmax(probs[0]) * 100).cpu().int().tolist()
        else:
            if torch.rand(1).item() < epsilon or self.total_steps < self.warmup_steps:
                if self.probs is not None:
                    return torch.multinomial(torch.tensor(self.probs), 1).item(), self.probs
                else:
                    return torch.randint(0, self.n_actions, (1,)).item(),  [0] * self.n_actions
            else:
                with torch.no_grad():
                    probs = self.model(state_tensor)
                    return torch.argmax(probs).item(), torch.round(F.softmax(probs[0]) * 100).cpu().int().tolist()

    def get_batch(self):
        transitions, indices, weights = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        self.logger.start_timer("state_action_reward_batch")
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.stack(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        self.logger.stop_timer("state_action_reward_batch")
        self.logger.start_timer("next_state_batch_final_mask_weights")
        non_final_next_states = [s for s in batch.next_state if s is not None]
        if non_final_next_states:
            next_state_batch = torch.cat(non_final_next_states).to(self.device)
        else:
            next_state_batch = torch.empty((0,) + state_batch.shape[1:], device=self.device)
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool, device=self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)
        self.logger.stop_timer("next_state_batch_final_mask_weights")
        return state_batch, action_batch, reward_batch, next_state_batch, non_final_mask, weights, indices

    def calc_loss(self):
        batch = self.logger.time_and_log(self.get_batch, "get_batch")
        state_batch, action_batch, reward_batch, next_state_batch, non_final_mask, weights, indices = batch
        state_action_values = self.model(state_batch).gather(1, action_batch)
        with torch.no_grad():
            next_state_values = torch.zeros(self.batch_size, device=self.device)
            next_state_values[non_final_mask] = self.target_net(next_state_batch).max(1)[0]
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)
        loss = (weights * F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1),
                                           reduction='none')).mean()
        self.debugger.track_loss(loss)
        return loss, expected_state_action_values, state_action_values, indices

    def apply_grad_clip(self):
        if self.clip_grad > 0:
                utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

    def optimize_model(self):
        if len(self.memory) < self.batch_size or len(self.memory) < self.warmup_steps:
            return
        loss, expected_state_action_values, state_action_values, indices = self.logger.time_and_log(self.calc_loss,
                                                                                                    "calc_loss")
        self.optimizer.zero_grad()
        loss.backward()
        self.debugger.track_gradients()
        self.apply_grad_clip()
        self.optimizer.step()
        with torch.no_grad():
            td_errors = (expected_state_action_values.unsqueeze(1) - state_action_values).abs().squeeze().cpu().numpy()
            self.memory.update_priorities(indices, td_errors)

    def check_failed_init(self, steps, done, episode, action, probs, debug=False):
        game_action = action
        if steps <= 1 and done:
            if debug:
                print("failed attempt")
                image = self.visualizer.save_current_frame(game_action, probs)
                plt.imshow(image)
                plt.show()
            if (episode + 1) % self.save_gif_every_x_epochs == 0:
                self.visualize_and_save_game_state(episode, game_action, probs)
            return True
        return False

    @staticmethod
    def get_next_state_tensor(terminated, obs):
        if terminated:
            return None
        if not isinstance(obs, torch.Tensor):
            return torch.FloatTensor(obs).unsqueeze(0)
        return obs.unsqueeze(0)

    def update_target_net(self):
        if len(self.memory) < self.batch_size or len(self.memory) < self.warmup_steps:
            return

        with torch.no_grad():
            for target_param, policy_param in zip(self.target_net.parameters(), self.model.parameters()):
                target_param.data.mul_(1 - self.TAU).add_(policy_param.data, alpha=self.TAU)

    def save_gif_if_needed(self, episode, action, probs):
        if (episode + 1) % self.save_gif_every_x_epochs == 0:
            self.visualize_and_save_game_state(
                episode,
                self.game_wrapper.preprocessor.postprocess_action(action),
                probs
            )

    def process_episode_step(self, steps, episode, action, probs, reward, terminated, truncated, state, obs):
        done = terminated or truncated
        if self.check_failed_init(steps, done, episode, action, probs):
            return done, True
        next_state_tensor = self.get_next_state_tensor(terminated, obs)
        self.memory.push(torch.tensor(state).to(torch.float32),
                         torch.tensor([action]),
                         next_state_tensor,
                         torch.tensor([reward]))
        if (self.total_steps + 1) % self.update_every_n_steps == 0:
            self.logger.time_and_log(self.optimize_model, "optimize_model")
        if (self.total_steps + 1) % self.update_target_every_n_steps == 0:
            self.logger.time_and_log(self.update_target_net, "update_target_net")
        self.logger.time_and_log(self.save_gif_if_needed, "save_gif_if_needed", *(episode, action, probs))
        self.total_steps += 1
        return done, False

    def run_episode(self, episode):
        state, info = self.logger.time_and_log(self.game_wrapper.reset, "reset_game", self.reset_options)
        done, steps, rewards = False, 0, []
        epsilon = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * episode / self.EPS_DECAY)
        while not done and steps <= self.max_episode_len:
            steps += 1
            action, probs = self.logger.time_and_log(self.choose_action, "choose_action", state, epsilon)
            obs, reward, terminated, truncated, _ = self.logger.time_and_log(self.game_wrapper.step, "g_step", action)
            ps_args = (steps, episode, action, probs, reward, terminated, truncated, state, obs)
            done, failed = self.logger.time_and_log(self.process_episode_step, "process_episode_step", *ps_args)
            if failed:
                break
            rewards.append(reward)
            state = obs
        score = self.game_wrapper.get_score()
        self.debugger.track_scores([score])
        self.debugger.track_epsilon(epsilon)
        self.game_wrapper.game.close()
        return np.sum(rewards), score

    def visualize_and_save_game_state(self, episode, game_action, probs):
        if (episode + 1) % self.save_gif_every_x_epochs == 0:
            image = self.visualizer.save_current_frame(game_action, probs)
            self.frames.append(image)

    def print_epoch_summary(self, episode, relevant_rewards, relevant_scores, validation=False):
        min_reward, max_reward = int(np.nanmin(relevant_rewards)), int(np.nanmax(relevant_rewards))
        mean_reward, mean_score = int(np.nanmean(relevant_rewards)), np.nanmean(relevant_scores)
        med_score, max_score = np.nanmedian(relevant_scores), np.nanmax(relevant_scores)
        if not validation:
            print(
                f"Episodes {episode + 1 - self.n_memory_episodes}-{episode + 1}: Min Reward: {min_reward:.2f},"
                f" Max Reward: {max_reward:.2f}, Mean Reward: {mean_reward:.2f}, Mean Score: {mean_score:.2f},"
                f" Median Score: {med_score:.2f}, Max Score: {max_score:.2f}")
        else:
            print(
                f"Episode {episode +1} Validation: Min Reward: {min_reward:.2f},"
                f" Max Reward: {max_reward:.2f}, Mean Reward: {mean_reward:.2f}, Mean Score: {mean_score:.2f},"
                f" Median Score: {med_score:.2f}, Max Score: {max_score:.2f}, N Validation Games: {len(relevant_scores)}")

    def log_and_compile_gif(self, episode):
        if (episode + 1) % self.save_gif_every_x_epochs == 0:
            gif_filename = f"{self.prefix_name}episode_{episode + 1}_score_{self.score_memory[-1]:.2f}.gif"
            imageio.mimsave(gif_filename, self.visualizer.pad_frames_to_same_size(self.frames), fps=self.fps, loop=0)
            print(f"GIF saved for episode {episode + 1}.")
            self.frames = []  # Clear frames after saving

        if (episode + 1) % self.n_memory_episodes == 0:
            relevant_rewards = self.rewards_memory[-self.n_memory_episodes:]
            relevant_scores = self.score_memory[-self.n_memory_episodes:]
            self.print_epoch_summary(episode, relevant_rewards, relevant_scores)
            self.rewards_memory = []

    def run_validation_move(self, steps, state, validation_episode, total_reward,
                            viz_total_reward, viz_score):
        action, probs = self.choose_action(state, validation_episode, True)
        obs, reward, terminated, truncated, _ = self.game_wrapper.step(action)
        done = terminated or truncated
        if self.check_failed_init(steps, done, -10 if validation_episode != 0 else -1, action, probs):
            return True, np.nan, self.game_wrapper.get_score(), obs, viz_total_reward, viz_score
        total_reward += reward
        if validation_episode == 0:
            self.visualize_and_save_game_state(self.save_gif_every_x_epochs - 1,
                                               self.game_wrapper.preprocessor.postprocess_action(action), probs)
            viz_total_reward, viz_score = total_reward, self.game_wrapper.get_score()
        return done, total_reward, self.game_wrapper.get_score(), obs, viz_total_reward, viz_score

    @staticmethod
    def pp_val_episode(scores, score, rewards, total_reward, validation_episode):
        scores.append(score)
        rewards.append(total_reward)
        print(" " * 100, end="\r")
        print(f"val episode: {validation_episode + 1}, current validation reward: {total_reward:.2f},"
              f" current score: {score:.2f}, n validation games: {len(scores)}", end="\r")
        return scores, rewards

    def save_validation_gif(self, episode, viz_score):
        gif_filename = f"{self.prefix_name}val_episode_{episode + 1}_score_{viz_score}.gif"
        imageio.mimsave(gif_filename, self.visualizer.pad_frames_to_same_size(self.frames), fps=self.fps, loop=0)
        print(f"GIF saved for episode {episode + 1}.")
        self.frames = []

    def validate_score(self, episode):
        rewards, scores, done, viz_total_reward, viz_score = [], [], False, 0, 0
        self.model.eval()
        for validation_episode in range(self.validate_episodes):
            state, info = self.game_wrapper.reset(options={"validation": True})
            score, total_reward, steps, done = 0, 0, 0, False
            with torch.no_grad():
                while not done and steps <= self.max_episode_len:
                    steps += 1
                    run_res = self.run_validation_move(steps, state, validation_episode, total_reward,
                                                       viz_total_reward, viz_score)
                    done, total_reward, score, state, viz_total_reward, viz_score = run_res
                scores, rewards = self.pp_val_episode(scores, score, rewards, total_reward, validation_episode)
                self.game_wrapper.game.close()
        self.on_validation_end(episode, rewards, scores, viz_score)

    def save_validation_csv(self):
        csv_filename = f"{self.prefix_name}validation_summary.csv"
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['episode', 'Mean Score', 'Median Score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for log_entry in self.validation_log:
                writer.writerow(log_entry)
        print(f"Validation summary saved to {csv_filename}")

    def plot_validation_convergence(self):
        episodes = [log_entry['episode'] for log_entry in self.validation_log]
        mean_scores = [log_entry['Mean Score'] for log_entry in self.validation_log]
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, mean_scores, marker='o', linestyle='-', color='b')
        plt.title('Validation Score Convergence')
        plt.xlabel('Episode')
        plt.ylabel('Mean Score')
        plt.grid(True)
        plot_filename = f"{self.prefix_name}validation_convergence.png"
        plt.savefig(plot_filename)
        plt.close()  # Close the figure to free memory
        print(f"Convergence plot saved to {plot_filename}")

    def on_validation_end(self, episode, rewards, scores, viz_score):
        self.print_epoch_summary(episode, rewards, scores, True)
        mean_score = np.nanmean(scores)
        self.game_wrapper.on_validation_end(episode, rewards, scores)
        self.model.train()
        self.save_validation_gif(episode, viz_score)
        if self.use_scheduler:
            self.scheduler.step(mean_score)
            for param_group in self.optimizer.param_groups:
                print("Current LR:", param_group['lr'])
        self.validation_log.append({"episode": episode+1, "Mean Score": mean_score, "Median Score": np.nanmedian(scores)})
        self.save_validation_csv()
        self.plot_validation_convergence()
        if ((episode+1)//self.validate_every_n_episodes) % self.save_model_every_n == 0:
            torch.save(self.model.state_dict(), f"{self.prefix_name}val_{episode+1}_score_{int(mean_score)}.pt")
        if self.best_model["score"] < mean_score:
            self.best_model = {"model": self.model, "score": mean_score}
            torch.save(self.model.state_dict(),
                       f"{self.prefix_name}best_model_{episode + 1}_score_{int(mean_score)}.pt")

    def train(self):
        for episode in range(self.episodes):
            total_reward, score = self.logger.time_and_log(self.run_episode, "run_episode", episode)
            if (episode + 1) % self.save_diagnostics == 0:
                self.debugger.plot_diagnostics(episode + 1)
                print(f"saved diagnostics episode: {episode + 1}")
            print(" " * 100, end="\r")
            print(f"current episode: {episode}, current reward: {total_reward}, current score: {score}", end="\r")
            if np.isnan(total_reward):
                print("debug")
            self.rewards_memory.append(total_reward)
            self.score_memory.append(score)
            self.logger.time_and_log(self.log_and_compile_gif, "log_and_compile_gif", episode)
            if (episode+1) % self.validate_every_n_episodes == 0:
                self.validate_score(episode)
                if self.early_stopping:
                    mean_scores = np.array([val_log["Mean Score"] for val_log in self.validation_log])
                    if np.all(max(mean_scores) > mean_scores[-min(self.early_stopping, len(mean_scores)):]):
                        print("early stopped")
                        break
                self.logger.dump_log("training_logs.csv")
