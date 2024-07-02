import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical
import numpy as np
from modeling.A2C import A2CAgent
import yaml

class PPOAgent(A2CAgent):
    def __init__(self, config_path, value_network, actor_network):
        super().__init__(config_path, value_network, actor_network)
        if isinstance(config_path, str):
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)['trainer']
        else:
            config = config_path

        self.clip_epsilon = float(config["clip_epsilon"])

    def training_batch(self, epochs, batch_size):
        actions, dones, rewards, values, observations, old_log_probs = self.initialize_batch_variables(batch_size)
        obs, info = self.game_wrapper.reset(self.reset_options)
        episode_count, total_reward, steps = 0, 0, 1
        for epoch in range(epochs):
            i = 0
            while i < batch_size:
                action, policy, value, obs_torch = self.get_action_and_value(obs)
                actions[i] = action
                values[i] = value
                observations[i] = obs_torch
                old_log_probs[i] = self.get_log_probs(policy, action)
                obs, reward, done = self.step_game(action)
                rewards[i], dones[i], total_reward = reward, done, total_reward + reward
                i, steps, episode_count, total_reward, obs = self.handle_episode_end(i, done, steps, total_reward,
                                                                                     episode_count, obs, policy, action)
            if self.check_early_stop():
                break
            self.update_model(observations, actions, rewards, dones, values, obs_torch, old_log_probs)
            self.save_diagnostics_if_needed(epoch)
        print(f'The training was done over a total of {episode_count} episodes')

    def initialize_batch_variables(self, batch_size):
        actions, dones = np.zeros((batch_size,), dtype=np.int32), np.zeros((batch_size,), dtype=bool)
        rewards, values = np.zeros((2, batch_size), dtype=np.float32)
        observations = torch.zeros((batch_size,) + self.input_shape, dtype=torch.float32).to(self.device)
        old_log_probs = np.zeros((batch_size,), dtype=np.float32)
        return actions, dones, rewards, values, observations, old_log_probs

    @staticmethod
    def get_log_probs(policy, action):
        log_probs = F.log_softmax(policy, dim=-1)
        action_log_prob = log_probs[0, action]
        return action_log_prob.cpu().detach().numpy()

    def optimize_model(self, observations, actions, returns, advantages, old_log_probs):
        actions, returns, advantages, observations, old_log_probs = self.prepare_tensors(actions, returns,
                                                                                         advantages, observations,
                                                                                         old_log_probs)
        value_loss = self.optimize_value_network(observations, returns)
        actor_loss, entropy = self.optimize_actor_network(observations, actions, advantages, old_log_probs)
        self.debugger.track_loss(actor_loss, value_loss)
        self.debugger.track_policy_entropy(entropy.cpu().detach().numpy())

    def prepare_tensors(self, actions, returns, advantages, observations, old_log_probs):
        actions = F.one_hot(torch.tensor(actions, dtype=torch.int64), self.n_actions).float().to(self.device)
        returns = torch.tensor(returns[:, None], dtype=torch.float).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float).to(self.device)
        observations = torch.tensor(observations, dtype=torch.float).to(self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float).to(self.device)
        return actions, returns, advantages, observations, old_log_probs

    def optimize_actor_network(self, observations, actions, advantages, old_log_probs):
        self.actor_optimizer.zero_grad()
        policies = self.actor_network(observations)
        actor_loss, entropy = self.compute_actor_loss_entropy(policies, actions, advantages, old_log_probs)
        actor_loss.backward()
        self.debugger.track_gradients(True)
        self.actor_optimizer.step()
        return actor_loss, entropy

    def compute_actor_loss_entropy(self, policies, actions, advantages, old_log_probs):
        probs, log_probs, log_action_probs = self.compute_probabilities(policies, actions)
        ratio = torch.exp(log_action_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        actor_loss = (-torch.min(surr1, surr2)).mean()
        entropy = -(probs * log_probs).sum(-1).mean()
        return actor_loss, entropy

    def update_model(self, observations, actions, rewards, dones, values, obs_torch, old_log_probs):
        next_value = [0] if dones[-1] else self.value_network(obs_torch).cpu().detach().numpy()[0]
        returns, advantages = self._returns_advantages(rewards, dones, values, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        self.optimize_model(observations, actions, returns, advantages, old_log_probs)
