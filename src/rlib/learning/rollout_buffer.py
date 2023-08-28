import numpy as np
import torch

class RolloutBuffer:

    def __init__(self, size, num_envs, state_space, action_space, batch_size, discount, use_gae, lambda_gae=None):

        self.size = size
        self.num_envs = num_envs
        self.state_space = state_space
        self.action_space = action_space
        self.batch_size = batch_size
        self.use_gae = use_gae
        self.discount=discount

        assert size % batch_size == 0, "The batch size should divide the size of the buffer."

        if use_gae:
            assert lambda_gae is not None, "Using GAE without specifying lambda"
            self.lambda_gae = lambda_gae
            
        self.reset()


    def store(self, action, state, reward, done, log_prob, value):

        if self.ptr >= self.size:
            raise ValueError("RolloutBuffer is already full")
        
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)

        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, dtype=torch.float32)

        if not isinstance(done, torch.Tensor):
            done = torch.tensor(done)

        if not isinstance(log_prob, torch.Tensor):
            log_prob = torch.tensor(log_prob, dtype=torch.float32)

        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32)

        self.actions[self.ptr] = action
        self.states[self.ptr] = state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value

        self.ptr += 1

    def compute_advantages(self, next_done, next_value):

        if not isinstance(next_done, torch.Tensor):
            next_done = torch.tensor(next_done).to(torch.int64)

        if not isinstance(next_value, torch.Tensor):
            next_value = torch.tensor(next_value, dtype=torch.float32)

        if self.ptr != self.size:
            raise ValueError("Trying to compute the advantages while the RolloutBuffer is not full.")

        self.returns = torch.zeros_like(self.rewards)

        if self.use_gae:

            self.advantages = torch.zeros_like(self.rewards)
            last_gae_lambda = 0

            for t in reversed(range(self.size)):

                if t == self.size - 1:
                    next_non_terminal = 1 - next_done
                    next_value_t = next_value

                else:
                    next_non_terminal = 1 - self.dones[t+1]
                    next_value_t = self.values[t+1]

                delta = self.rewards[t] + self.discount * next_value_t * next_non_terminal - self.values[t]
                self.advantages[t] = last_gae_lambda = delta + self.discount * self.lambda_gae * next_non_terminal * last_gae_lambda
            
            self.returns = self.advantages + self.values

        else:

            for t in reversed(range(self.size)):

                if t == self.size - 1:
                    next_non_terminal = 1 - next_done
                    next_return = next_value

                else:
                    next_non_terminal = 1 - self.dones[t+1]
                    next_return = self.returns[t+1]
                
                self.returns[t] = self.rewards[t] + self.discount * next_non_terminal * next_return

            self.advantages = self.returns - self.values

    def batches(self):

        permuted_indices = np.random.permutation(np.arange(self.size))

        self.actions = self.actions.reshape((-1,) + self.action_space.shape)
        self.states = self.states.reshape((-1,) + self.state_space.shape)
        self.log_probs = self.log_probs.reshape(-1)
        self.advantages = self.advantages.reshape(-1)
        self.returns = self.returns.reshape(-1)
        self.values = self.values.reshape(-1)

        for i in range(self.size // self.batch_size):

            start = i * self.batch_size
            end = (i+1) * self.batch_size

            batch_indices = permuted_indices[start:end]

            yield(
                self.actions[batch_indices],
                self.states[batch_indices],
                self.log_probs[batch_indices],
                self.advantages[batch_indices],
                self.returns[batch_indices],
                self.values[batch_indices]
            )

    def reset(self):

        self.actions = torch.zeros(self.size, self.num_envs, *self.action_space.shape, dtype=torch.float32)
        self.states = torch.zeros(self.size, self.num_envs, *self.state_space.shape, dtype=torch.float32)
        self.rewards = torch.zeros(self.size, self.num_envs, dtype=torch.float32)
        self.dones = torch.zeros(self.size, self.num_envs)
        self.log_probs = torch.zeros(self.size, self.num_envs, dtype=torch.float32)
        self.values = torch.zeros(self.size, self.num_envs, dtype=torch.float32)

        self.ptr = 0
