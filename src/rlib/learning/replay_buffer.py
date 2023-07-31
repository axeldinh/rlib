
import numpy as np

class ReplayBuffer:

    def __init__(self, max_size):

        self.max_size = max_size
        self.position = 0
        self.size = 0

    def store(self, s, a, r, s2, d):

        if self.size == 0:

            num_obs = len(s.reshape(-1))
            num_actions = len(a.reshape(-1))

            self.states = np.zeros((self.max_size, num_obs), dtype=np.float32)
            self.actions = np.zeros((self.max_size, num_actions), dtype=np.float32)
            self.rewards = np.zeros((self.max_size, 1), dtype=np.float32)
            self.next_states = np.zeros((self.max_size, num_obs), dtype=np.float32)
            self.dones = np.zeros((self.max_size, 1), dtype=np.float32)

        self.states[self.position] = s
        self.actions[self.position] = a
        self.rewards[self.position] = r
        self.next_states[self.position] = s2
        self.dones[self.position] = d

        self.position = (self.position + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):

        indices = np.random.randint(0, self.size, size=batch_size)

        return self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices], self.dones[indices]

    def __len__(self):

        return self.size