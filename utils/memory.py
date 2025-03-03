import random
import numpy as np
from collections import deque


class ReplayMemory:
    """
    Experience Replay Buffer for DQN training.
    Stores transitions and allows random sampling for batch training.
    """

    def __init__(self, capacity):
        """
        Initialize replay buffer with fixed capacity.

        Args:
            capacity (int): Maximum number of transitions to store
        """
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Store a transition in the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether this transition ended the episode
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of transitions randomly.

        Args:
            batch_size (int): Number of transitions to sample

        Returns:
            list: Batch of transitions
        """
        # Ensure we don't sample more than are available
        batch_size = min(batch_size, len(self.memory))
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Get current buffer size.

        Returns:
            int: Number of transitions in buffer
        """
        return len(self.memory)


class PrioritizedReplayMemory:
    """
    Prioritized Experience Replay Buffer.
    Stores transitions with priorities and samples based on those priorities.
    Higher priority transitions are more likely to be sampled.
    """

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Initialize prioritized replay buffer with fixed capacity.

        Args:
            capacity (int): Maximum number of transitions to store
            alpha (float): Controls how much prioritization is used (0 = uniform, 1 = full prioritization)
            beta_start (float): Initial value of beta for importance sampling correction
            beta_frames (int): Number of frames over which beta will be annealed to 1
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """
        Store a transition in the buffer with maximum priority.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether this transition ended the episode
        """
        max_priority = self.priorities.max() if self.memory else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.memory[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample a batch of transitions based on priorities.

        Args:
            batch_size (int): Number of transitions to sample

        Returns:
            tuple: (batch of transitions, importance sampling weights, indices)
        """
        if len(self.memory) == 0:
            return []

        # Calculate beta for importance sampling
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        # Calculate sampling probabilities
        if len(self.memory) < self.capacity:
            probs = self.priorities[:len(self.memory)]
        else:
            probs = self.priorities

        probs = probs ** self.alpha
        probs = probs / probs.sum()

        # Sample transitions
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        # Calculate importance sampling weights
        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights = weights / weights.max()

        return samples, weights, indices

    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled transitions.

        Args:
            indices (list): Indices of transitions to update
            priorities (list): New priorities for those transitions
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        """
        Get current buffer size.

        Returns:
            int: Number of transitions in buffer
        """
        return len(self.memory)