import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import yaml
import imageio
import os
import cv2
from collections import deque, namedtuple
from torch.nn import functional as F
from gym import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
from modeling.AtariGameWrapper import AtariGameWrapper, AtariGameViz
from modeling.models import CNNDQNAgent, DQN, ActorCritic


class CustomAtariEnv(gym.Env):
    def __init__(self, config):
        super(CustomAtariEnv, self).__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.game = gym.make(self.config['trainer']['game'], render_mode=self.config['trainer']['render_mode'])
        self.game_wrapper = AtariGameWrapper(self.game, self.config['atari_game_wrapper'])
        self.visualizer = AtariGameViz(self.game, self.device)
        obs, _ = self.game_wrapper.reset()
        obs_shape = obs.shape
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = self.game.action_space

    def reset(self):
        obs, _ = self.game_wrapper.reset()
        return obs.cpu().numpy()

    def step(self, action):
        obs, reward, done, truncated, info = self.game_wrapper.step(action)
        # Save frame for visualization
        self.visualizer.save_current_frame(action, [])
        return obs.cpu().numpy(), reward, done or truncated, info

    def render(self, mode='human'):
        return self.game_wrapper.game.render(mode=mode)

    def close(self):
        self.game_wrapper.game.close()

# Custom policy using your models
class CustomDQNPolicy(BasePolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(CustomDQNPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.net = CNNDQNAgent(
            input_shape=observation_space.shape,
            output_size=action_space.n,
            conv_layers_params=kwargs.get('conv_layers_params'),
            fc_layers=kwargs.get('fc_layers'),
            dueling=kwargs.get('dueling')
        ).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr_schedule(1))

    def forward(self, obs, deterministic=True):
        obs = torch.as_tensor(obs).float().to(self.device)
        if len(obs.shape) == len(self.observation_space.shape):
            obs = obs.unsqueeze(0)
        q_values = self.net(obs)
        return q_values

    def _predict(self, observation, deterministic=True):
        q_values = self.forward(observation)
        actions = q_values.argmax(dim=1)
        return actions

    def predict(self, observation, state=None, mask=None, deterministic=True):
        actions = self._predict(observation, deterministic)
        return actions.cpu().numpy(), state

# Custom callback to integrate logging, validation, and visualization
class CustomCallback(BaseCallback):
    def __init__(self, config, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.config = config
        self.visualizer = None  # Will be initialized in _on_training_start
        self.episodes = 0
        self.total_steps = 0
        self.rewards_memory = []
        self.score_memory = []
        self.n_memory_episodes = int(config['trainer']['n_memory_episodes'])
        self.save_gif_every_x_epochs = int(config['trainer']['save_gif_every_x_epochs'])
        self.validate_every_n_episodes = int(config['trainer']['validate_every_n_episodes'])
        self.validate_episodes = int(config['trainer']['validate_episodes'])
        self.frames = []
        self.fps = int(config['trainer']['gif_fps'])
        self.prefix_name = os.path.join(config['trainer']['folder'], config['trainer']['prefix_name'])
        os.makedirs(config['trainer']['folder'], exist_ok=True)
        self.validation_log = []
        self.best_mean_reward = -np.inf
        self.early_stopping = config['trainer'].get('early_stopping', 0)
        self.episodes_since_improvement = 0

    def _on_training_start(self):
        self.visualizer = self.training_env.envs[0].visualizer

    def _on_step(self):
        self.total_steps += 1
        done = self.locals['dones'][0]
        reward = self.locals['rewards'][0]
        info = self.locals['infos'][0]
        if not hasattr(self, 'current_rewards'):
            self.current_rewards = []
        self.current_rewards.append(reward)
        if done:
            episode_reward = sum(self.current_rewards)
            self.rewards_memory.append(episode_reward)
            self.current_rewards = []
            self.episodes += 1
            if (self.episodes) % self.save_gif_every_x_epochs == 0:
                self.save_gif(self.episodes)
            if (self.episodes) % self.validate_every_n_episodes == 0:
                self.validate(self.episodes)
            if self.early_stopping > 0:
                mean_validation_scores = [log['Mean Score'] for log in self.validation_log]
                if len(mean_validation_scores) >= self.early_stopping:
                    recent_scores = mean_validation_scores[-self.early_stopping:]
                    if all(score <= self.best_mean_reward for score in recent_scores):
                        print("Early stopping due to no improvement in validation scores.")
                        return False
        return True

    def save_gif(self, episode):
        gif_filename = f"{self.prefix_name}_episode_{episode}.gif"
        frames = self.visualizer.frames
        if frames:
            imageio.mimsave(gif_filename, frames, fps=self.fps)
            print(f"Saved GIF for episode {episode} to {gif_filename}")
            self.visualizer.frames = []
        else:
            print(f"No frames to save for episode {episode}")

    def validate(self, episode):
        total_rewards = []
        total_scores = []
        env = self.training_env.envs[0]
        for _ in range(self.validate_episodes):
            obs = env.reset()
            done = False
            rewards = []
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                rewards.append(reward)
            total_rewards.append(sum(rewards))
            total_scores.append(env.game_wrapper.get_score())
        mean_reward = np.mean(total_rewards)
        mean_score = np.mean(total_scores)
        self.validation_log.append({'episode': episode, 'Mean Reward': mean_reward, 'Mean Score': mean_score})
        print(f"\nValidation after episode {episode}: Mean Reward = {mean_reward}, Mean Score = {mean_score}")
        if mean_score > self.best_mean_reward:
            self.best_mean_reward = mean_score
            model_filename = f"{self.prefix_name}_best_model_episode_{episode}_score_{int(mean_score)}"
            self.model.save(model_filename)
            print(f"Saved new best model to {model_filename}")

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


config_path = os.path.join("modeling", "configs", "trainer_config_CarRacing_lr1e4.yaml")

config = load_config(config_path)
# Create the environment
env = CustomAtariEnv(config)

# Define the learning rate schedule
def lr_schedule(_):
    return config['trainer']['learning_rate']

# Define policy kwargs
policy_kwargs = {
    'conv_layers_params': config['conv_layers_params'],
    'fc_layers': config['fc_layers'],
    'dueling': config['trainer'].get('use_ddqn', False)
}

# Create the model
model = DQN(
    policy=CustomDQNPolicy,
    env=env,
    learning_rate=lr_schedule,
    gamma=config['trainer']['gamma'],
    batch_size=config['trainer']['batch_size'],
    buffer_size=config['trainer']['replaymemory'],
    learning_starts=config['trainer']['warmup_steps'],
    target_update_interval=config['trainer']['update_target_every_n_steps'],
    train_freq=config['trainer']['update_every_n_steps'],
    exploration_initial_eps=config['trainer']['EPS_START'],
    exploration_final_eps=config['trainer']['EPS_END'],
    exploration_fraction=config['trainer']['EPS_DECAY'] / (config['trainer']['episodes'] * config['trainer']['max_episode_len']),
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=config['trainer']['folder']
)

# Create the custom callback
callback = CustomCallback(config)

# Calculate total timesteps
total_timesteps = config['trainer']['episodes'] * config['trainer']['max_episode_len']

# Start training
model.learn(
    total_timesteps=total_timesteps,
    callback=callback
)