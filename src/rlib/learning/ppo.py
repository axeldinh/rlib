import numpy as np
import torch
from copy import deepcopy

from rlib.learning import BaseAlgorithm
from rlib.agents import get_agent

class PPOAgent(torch.nn.Module):

    def __init__(self, policy, advantage):

        super().__init__()
        
        self.policy_agent = policy
        self.old_policy_agent = deepcopy(policy)
        self.advantage = advantage

    def policy(self, state):

        logits = self.policy_agent(state)
        return torch.nn.functional.softmax(logits, dim=-1)

    def old_policy(self, state):
            
        logits = self.old_policy_agent(state)
        return torch.nn.functional.softmax(logits, dim=-1)

    def get_action(self, state):

        action = self.policy(state)
        return torch.argmax(action).numpy()



class PPO(BaseAlgorithm):

    def __init__(self, env_kwargs, policy_kwargs, advantage_kwargs, discount=0.99,
                 num_iterations=100000, epsilon=0.2, test_every_n_steps=1000,
                 update_every_n_steps=500, policy_lr=3e-4, advantage_lr=3e-4,
                 max_episode_length=-1, max_total_reward=-1, save_folder="ppo",
                 normalize_observations=False):
        """
        :param env_kwargs: Keyword arguments to call `gym.make(**env_kwargs, render_mode=render_mode)`
        :type env_kwargs: dict
        :param policy_kwargs: Keyword arguments to call `get_agent(agent_type, **policy_kwargs)`
        :type policy_kwargs: dict
        :param advantage_kwargs: Keyword arguments to call `get_agent(agent_type, **advantage_kwargs)`
        :type advantage_kwargs: dict
        :param discount: Discount factor. Defaults to 0.99.
        :type discount: float
        :param update_every_n_steps: Number of steps to collect before updating the policy. Defaults to 1000.
        :type update_every_n_steps: int
        :param policy_lr: Learning rate for the policy. Defaults to 3e-4.
        :type policy_lr: float
        :param advantage_lr: Learning rate for the advantage. Defaults to 3e-4.
        :type advantage_lr: float
        :param max_episode_length: Maximum length of an episode. Defaults to -1 (no limit).
        :type max_episode_length: int
        :param max_total_reward: Maximum total reward of an episode. Defaults to -1 (no limit).
        :type max_total_reward: float
        :param save_folder: Folder to save the model. Defaults to "ppo".
        :type save_folder: str
        :param normalize_observations: Whether to normalize observations in `[-1, 1]`. Defaults to False.
        :type normalize_observations: bool

        """

        super().__init__(env_kwargs, max_episode_length, max_total_reward, 
                       save_folder, normalize_observations)
        
        self.policy_kwargs = policy_kwargs
        self.advantage_kwargs = advantage_kwargs
        self.policy_lr = policy_lr
        self.advantage_lr = advantage_lr
        self.discount = discount
        self.epsilon = epsilon
        self.num_iterations = num_iterations
        self.update_every_n_steps = update_every_n_steps

        # Build the actor and critic

        if self.obs_shape.__len__() == 1:

            num_obs = self.obs_shape[0]
            num_actions = self.action_shape[0]

            policy_kwargs["input_size"] = num_obs
            policy_kwargs["output_size"] = num_actions
            policy_kwargs['requires_grad'] = True

            advantage_kwargs["input_size"] = num_obs
            advantage_kwargs["output_size"] = 1
            advantage_kwargs['requires_grad'] = True        
            self.policy = get_agent("mlp", **policy_kwargs)
            self.advantage = get_agent("mlp", **advantage_kwargs)

        else:
            raise NotImplementedError("PPO only supports 1D observations currently.")
        
        self.agent = PPOAgent(self.policy, self.advantage)
        self.old_agent = deepcopy(self.agent)
        
        self.policy_optimizer = torch.optim.Adam(self.agent.policy_agent.parameters(), lr=self.policy_lr, maximize=True)
        self.advantage_optimizer = torch.optim.Adam(self.agent.advantage.parameters(), lr=self.advantage_lr)

        self.current_iteration = 0
        self.current_episode = 0

        self.test_rewards = []


    def train(self):

        # First init the env

        env = self.make_env()

        while self.current_iteration < self.num_iterations:

            # Collect the trajectories
            s, a, r, s_, d, iters, episodes = self.collect_trajectories(env)

            # Update the policy
            self.update_networks(s, a, r, s_, d)
            
            # Update the current iteration and episode
            self.current_iteration += iters
            self.current_episode += episodes

            mean , std = self.test(10)
            
            print(f"Iteration: {self.current_iteration} | Episode: {self.current_episode} | Reward: {mean} | Std: {std}")

        input("Press any key for last test...")
        mean, std = self.test(10, display=True)
        print(f"Mean: {mean} | Std: {std}")

    def collect_trajectories(self, env):

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        state, _ = env.reset()
        done = False
        iter_ = 0
        episode = 0

        while iter_ < self.update_every_n_steps:

            state, _ = env.reset()
            done = False

            while not done:

                # Get the action
                state = torch.tensor(state, dtype=torch.float32)
                action = self.agent.get_action(state)

                # Step the environment
                next_state, reward, done, _, _ = env.step(action)

                iter_ += 1

                if iter_ >= self.max_episode_length:
                    done = True

                # Append the data
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

                # Update the current state
                state = next_state


            episode += 1

        return states, actions, rewards, next_states, dones, iter_, episode

    def update_networks(self, states, actions, rewards, next_states, dones):

        # Convert everything to tensors
        states = torch.tensor(np.stack(states))
        actions = torch.tensor(np.stack(actions))
        rewards = torch.tensor(np.stack(rewards)).to(torch.float32)
        next_states = torch.tensor(np.stack(next_states))
        dones = torch.tensor(np.stack(dones))
        
        # Get the indexes of the dones
        dones_idx = torch.where(dones == True)[0]
        num_episodes = dones_idx.shape[0]

        returns = torch.zeros_like(rewards)

        # Compute the returns
        for i in range(num_episodes):
            returns[dones_idx[i]] = rewards[dones_idx[i]]
            for j in range(dones_idx[i] - 1, dones_idx[i-1] if i-1 >= 0 else -1, -1):
                returns[j] = rewards[j] + self.discount * returns[j + 1]

        # Normalize the returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute the advantages
        advantages = returns.unsqueeze(-1) - self.old_agent.advantage(states)

        # Compute the old log probabilities
        old_log_probs = torch.log(self.old_agent.policy(states))
        log_probs = torch.log(self.agent.policy(states))
        
        r = torch.exp(log_probs - old_log_probs)
        surrogate1 = r * advantages
        surrogate2 = torch.clamp(r, 1 - self.epsilon, 1 + self.epsilon) * advantages

        # Compute the loss
        policy_loss = torch.min(surrogate1, surrogate2).mean()
        advantage_loss = torch.nn.functional.mse_loss(advantages, self.agent.advantage(states))

        # Update the old policy BEFORE updating the policy
        for p, p_old in zip(self.agent.parameters(), self.old_agent.parameters()):
            p_old.data.copy_(p.data)

        # Update the networks
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        self.advantage_optimizer.zero_grad()
        advantage_loss.backward()
        self.advantage_optimizer.step()



if __name__ == "__main__":

    env_kwargs = {"id": "CartPole-v1"}
    policy_kwargs = {"hidden_sizes": [64, 64]}
    advantage_kwargs = {"hidden_sizes": [64, 64]}

    ppo = PPO(env_kwargs, policy_kwargs, advantage_kwargs, update_every_n_steps=500, num_iterations=1_000_000, discount=0.99, epsilon=0.2, max_episode_length=500)
    ppo.train()