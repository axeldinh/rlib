import numpy as np
import torch
from torch.distributions import Categorical, Normal
from copy import deepcopy

from rlib.learning import BaseAlgorithm
from rlib.agents import get_agent

class PPOAgent(torch.nn.Module):

    def __init__(self, distribution, policy_agent, value_agent):

        super().__init__()

        """
        self.shared_layers = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU()
        )

        self.policy_layers = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2)
        )

        self.value_layers = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

        self.policy_agent = torch.nn.Sequential(self.shared_layers, self.policy_layers)
        self.value_agent = torch.nn.Sequential(self.shared_layers, self.value_layers)
        """
        
        self.policy_agent = policy_agent
        self.value_agent = value_agent

        self.distribution = distribution

        if distribution == "categorical":
            self.distribution = Categorical
        elif distribution == "normal":
            self.distribution = Normal
            self.std = None

    def policy(self, state):
        # Returns the logits of the policy
        return self.policy_agent(state)
    
    def value(self, state):
        # Returns the value of the state
        return self.value_agent(state)
    
    def get_action(self, state, deterministic=True):

        is_np = isinstance(state, np.ndarray)

        if is_np:
            state = torch.tensor(state, dtype=torch.float32)

        dist = self.get_distribution(state)
        if deterministic:
            if isinstance(dist, Normal):
                action = dist.mean
            elif isinstance(dist, Categorical):
                action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()

        if is_np:
            action = action.detach().numpy()
        
        return action
    
    def forward(self, state):
        
        dist = self.get_distribution(state)
        value = self.value(state)

        return dist, value

    def get_distribution(self, state):

        out = self.policy(state)

        if self.distribution == Normal:
            if self.std is None:
                self.std = torch.nn.Parameter(torch.ones_like(out), requires_grad=True)
            dist = self.distribution(out, self.std)
        else:
            dist = self.distribution(logits=out)

        return dist



class PPO(BaseAlgorithm):

    def __init__(self, env_kwargs, policy_kwargs, value_kwargs, discount=0.99, gae_lambda=0.95,
                 normalize_advantage=True, value_coef=0.5,
                 num_iterations=100000, epsilon=0.2, test_every_n_steps=1000,
                 update_every_n_steps=500, learning_rate=3e-4,
                 update_lr_fn=None, batch_size=64, n_updates=10,
                 max_grad_norm=0.5, target_kl=None,
                 max_episode_length=-1, max_total_reward=-1, save_folder="ppo",
                 normalize_observations=False):
        """
        :param env_kwargs: Keyword arguments to call `gym.make(**env_kwargs, render_mode=render_mode)`
        :type env_kwargs: dict
        :param policy_kwargs: Keyword arguments to call `get_agent(agent_type, **policy_kwargs)`
        :type policy_kwargs: dict
        :param value_kwargs: Keyword arguments to call `get_agent(agent_type, **value_kwargs)`
        :type value_kwargs: dict
        :param discount: Discount factor. Defaults to 0.99.
        :type discount: float
        :param gae_lambda: Lambda for the generalized advantage estimation. Defaults to 0.95.
        :type gae_lambda: float
        :param normalize_advantage: Whether to normalize the advantage. Defaults to True.
        :type normalize_advantage: bool
        :param value_coef: Coefficient for the value loss. Defaults to 0.5.
        :type value_coef: float
        :param target_kl: Target KL divergence. Defaults to None (No limit).
        :type target_kl: float
        :param update_every_n_steps: Number of steps to collect before updating the policy. Defaults to 1000.
        :type update_every_n_steps: int
        :param learning_rate: Learning rate for the policy and value function. Defaults to 3e-4.
        :type learning_rate: float
        :param update_lr_fn: Function to update the learning rate. Should take in the current iteration and return two floats (policy_lr, advantage_lr). Defaults to None (No update).
        :type update_lr_fn: function
        :param batch_size: Batch size for training. Defaults to 64.
        :type batch_size: int
        :param n_updates: Number of updates to perform after each rollout. Defaults to 10.
        :type n_updates: int
        :param max_grad_norm: Maximum gradient norm. Defaults to 0.5.
        :type max_grad_norm: float
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
        self.value_kwargs = value_kwargs
        self.learning_rate = learning_rate
        self.update_lr_fn = update_lr_fn
        self.batch_size = batch_size
        self.n_updates = n_updates
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.normalize_advantage = normalize_advantage
        self.epsilon = epsilon
        self.num_iterations = num_iterations
        self.update_every_n_steps = update_every_n_steps
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.value_coef = value_coef

        # Build the actor and critic

        if self.obs_shape.__len__() == 1:

            num_obs = self.obs_shape[0]
            num_actions = self.action_shape[0]

            policy_kwargs["input_size"] = num_obs
            policy_kwargs["output_size"] = num_actions
            policy_kwargs['requires_grad'] = True

            value_kwargs["input_size"] = num_obs
            value_kwargs["output_size"] = 1
            value_kwargs['requires_grad'] = True        
            self.policy = get_agent("mlp", **policy_kwargs)
            self.advantage = get_agent("mlp", **value_kwargs)

        else:
            raise NotImplementedError("PPO only supports 1D observations currently.")
        
        if self.action_space_type == "discrete":
            self.distribution = "categorical"
        elif self.action_space_type == "continuous":
            self.distribution = "normal"
        
        self.current_agent = PPOAgent(self.distribution, self.policy, self.advantage)
        
        self.optimizer = torch.optim.Adam(self.current_agent.policy_agent.parameters(), lr=self.learning_rate)

        self.current_iteration = 0
        self.current_episode = 0

        self.test_rewards = []
        self.losses = []

    def train(self):

        env = self.make_env()

        while self.current_iteration < self.num_iterations:

            states, actions, returns, gaes, log_probs = self.rollout(env)

            states = torch.tensor(np.stack(states), dtype=torch.float32)
            actions = torch.stack(actions).to(torch.int64)
            returns = torch.tensor(returns, dtype=torch.float32)
            gaes = torch.tensor(gaes, dtype=torch.float32)
            log_probs = torch.stack(log_probs)

            if self.normalize_advantage:
                gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

            policy_loss, value_loss = self.update_networks(states, actions, log_probs, gaes, returns)

            self.losses.append({"policy_loss": policy_loss, "advantage_loss": value_loss})

            reward, std = self.test(1)
            print(f"Iteration: {self.current_iteration} | Policy Loss: {policy_loss:.2e} | Value Loss: {value_loss:.2e} | Reward: {reward:.2f} (+/- {std:.2f})")


    def rollout(self, env):

        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []

        state, _ = env.reset()
        done = False
        iters = 0
        episode_length = 0
        episode_reward = 0

        while True:
            
            dist, value = self.current_agent(torch.tensor(state))
            action = dist.sample().detach()
            log_prob = dist.log_prob(action).detach()
            value = value.detach().item()

            next_state, reward, done, _, _ = env.step(action.numpy())

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            values.append(value)
            log_probs.append(log_prob)

            state = next_state
            iters += 1

            episode_length += 1
            episode_reward += reward

            if episode_length >= self.max_episode_length or episode_reward >= self.max_total_reward:
                done = True

            if done:
                state, _ = env.reset()
                done = False
                self.current_episode += 1

                if iters >= self.update_every_n_steps:
                    break

        self.current_iteration += iters

        # Compute the returns and advantages
        returns = np.zeros(len(rewards))
        gaes = np.zeros(len(rewards))

        # Get the slices of the episodes
        dones_idx = np.where(dones)[0]

        for i in range(len(dones_idx)):

            slice_ = slice(dones_idx[i-1] if i > 0 else 0, dones_idx[i] + 1)
            returns[slice_] = self.compute_returns_one_episode(rewards[slice_])
            gaes[slice_] = self.compute_gaes_one_episode(rewards[slice_], values[slice_])

        return states, actions, returns, gaes, log_probs

    def compute_returns_one_episode(self, rewards):
        
        return_ = rewards[-1]
        returns = [return_]

        for i in reversed(range(len(rewards)-1)):
            return_ = rewards[i] + self.discount * return_
            returns.insert(0, return_)

        return returns
    
    def compute_gaes_one_episode(self, rewards, values):

        nex_values = np.concatenate([values[1:], [0]])
        deltas = [reward + self.discount * nex_val - val for reward, val, nex_val in zip(rewards, values, nex_values)]

        gaes = [deltas[-1]]
        for i in reversed(range(len(deltas)-1)):
            gaes.append(deltas[i] + self.discount * self.gae_lambda * gaes[-1])
            
        return gaes[::-1]
    
    def update_networks(self, states, actions, log_probs, gaes, returns):

        # Make batches of size `batch_size`
        permuted_indices = np.random.permutation(len(states))
        states = states[permuted_indices]
        actions = actions[permuted_indices]
        log_probs = log_probs[permuted_indices]
        gaes = gaes[permuted_indices].unsqueeze(-1)
        returns = returns[permuted_indices]

        batch_indexes = np.arange(len(states))
        num_batches = len(states) // self.batch_size
        if len(states) % self.batch_size != 0:
            num_batches += 1

        for _ in range(self.n_updates):

            np.random.shuffle(batch_indexes)

            for i in range(num_batches):

                batch_idx = batch_indexes[i*self.batch_size:(i+1)*self.batch_size]

                self.optimizer.zero_grad()

                new_dist = self.current_agent.get_distribution(states[batch_idx])
                new_log_probs = new_dist.log_prob(actions[batch_idx])

                ratio = torch.exp(new_log_probs - log_probs[batch_idx])
                clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                clipped_loss = torch.min(ratio * gaes[batch_idx], clipped_ratio * gaes[batch_idx])

                policy_loss = torch.mean(clipped_loss)

                value_loss = torch.mean((self.current_agent.value(states[batch_idx]) - returns) ** 2)

                loss = -policy_loss + value_loss * self.value_coef

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.current_agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = new_log_probs - log_probs[batch_idx]
                    kl_divergence = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                if self.target_kl is not None:
                    if kl_divergence > 1.5 * self.target_kl:
                        break

        return policy_loss.item(), value_loss.item()

    def update_policy(self, states, actions, log_probs, gaes):

        for _ in range(self.n_updates):

            self.optimizer.zero_grad()

            new_logits = self.current_agent.policy(states)
            new_distribution = Categorical(logits=new_logits)
            new_log_probs = new_distribution.log_prob(actions)

            ratio = torch.exp(new_log_probs - log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            clipped_loss = torch.min(ratio * gaes, clipped_ratio * gaes)

            policy_loss = torch.mean(clipped_loss)
            policy_loss.backward()
            self.optimizer.step()

            kl_divergence = torch.mean(log_probs - new_log_probs)
            if torch.abs(kl_divergence) > 0.01:
                break

        return policy_loss.item()
    
    def update_value(self, states, returns):

        for _ in range(self.n_updates):

            self.value_optimizer.zero_grad()

            values = self.current_agent.value(states)
            value_loss = torch.mean((values - returns) ** 2)
            value_loss.backward()
            self.value_optimizer.step()

        return value_loss.item()


if __name__ == "__main__":

    #env_kwargs = {"id": "CartPole-v1"}
    env_kwargs = {"id": "BipedalWalker-v3"}
    policy_kwargs = {"hidden_sizes": [64, 64]}
    advantage_kwargs = {"hidden_sizes": [64, 64]}

    num_iterations = 1_000_000

    ppo = PPO(env_kwargs, policy_kwargs, advantage_kwargs, update_every_n_steps=2048, num_iterations=num_iterations, discount=0.99, epsilon=0.2, max_episode_length=300,
              learning_rate=3e-4, n_updates=10, batch_size=64)
    ppo.train()

    import matplotlib.pyplot as plt

    losses = ppo.losses
    policy_losses = [loss["policy_loss"] for loss in losses]
    advantage_losses = [loss["advantage_loss"] for loss in losses]

    plt.plot(policy_losses, label="Policy Loss")
    plt.legend()
    plt.show()

    plt.plot(advantage_losses, label="Advantage Loss")
    plt.yscale('log')
    plt.legend()
    plt.show()

    input("Press any key for last test...")
    mean, std = ppo.test(10, display=True)
    print(f"Mean: {mean} | Std: {std}")
    """

    from stable_baselines3 import PPO
    import gymnasium as gym

    env = gym.make("CartPole-v1", render_mode="rgb_array")

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000)

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):

        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, _ = vec_env.step(action)
        env.render()
        if dones:
            obs = env.reset()
        vec_env.render('human')

        if dones:
            break
            """