import random
from collections import namedtuple, deque
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PERMemory(object):
    def __init__(self, capacity, alpha=0.6):
        self.alpha = alpha
        self.capacity = capacity
        self.memory = np.array([None] * capacity, dtype=object)
        self.priorities = np.array([0.0] * capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(self, *args):
        """Save a transition with priority."""
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        self.memory[self.position] = Transition(*args)
        self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size, beta=0.4):
        """Sample a batch of transitions based on priorities."""
        if self.size == self.capacity:
            probabilities = self.priorities ** self.alpha
        else:
            probabilities = self.priorities[:self.size] ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(self.size, batch_size, p=probabilities)
        samples = self.memory[indices]

        total = self.size
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return list(samples), indices, weights

    def update_priorities(self, indices, errors, offset=0.01):
        """Update priorities based on TD errors."""
        self.priorities[indices] = (np.abs(errors) + offset) ** self.alpha

    def __len__(self):
        return self.size