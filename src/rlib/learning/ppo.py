from datetime import timedelta
import time
import numpy as np
import torch
from torch.distributions import Categorical, Normal

from rlib.learning import BaseAlgorithm
from rlib.agents import get_agent

class PPOAgent(torch.nn.Module):

    def __init__(self, distribution, policy_agent, value_agent):

        super().__init__()

        """
        self.shared_layers = torch.nn.Sequential(
            torch.nn.Linear(policy_agent.layers[0].in_features, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU()
        )

        self.policy_layers = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, policy_agent.layers[-1].out_features)
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
            self.std = torch.nn.Parameter(torch.ones(1), requires_grad=True)

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
            dist = self.distribution(out, self.std)
        else:
            dist = self.distribution(logits=out)

        return dist



class PPO(BaseAlgorithm):

    def __init__(self, env_kwargs, policy_kwargs, value_kwargs, discount=0.99, gae_lambda=0.95,
                 normalize_advantage=True, value_coef=0.5,
                 num_iterations=100000, epsilon=0.2, test_every_n_steps=1000,
                 num_test_episodes=10,
                 update_every_n_steps=500, learning_rate=3e-4,
                 update_lr_fn=None, batch_size=64, n_updates=10,
                 max_grad_norm=0.5, target_kl=None,
                 max_episode_length=-1, max_total_reward=-1, save_folder="ppo",
                 normalize_observation=False):
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
        :param normalize_observation: Whether to normalize observations in `[-1, 1]`. Defaults to False.
        :type normalize_observation: bool

        """

        super().__init__(env_kwargs, max_episode_length, max_total_reward, 
                       save_folder, normalize_observation)
        
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
        self.test_every_n_steps = test_every_n_steps
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.value_coef = value_coef
        self.num_test_episodes = num_test_episodes

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

        self.test_mean_rewards = []
        self.test_std_rewards = []
        self.losses = []

        self.next_test = self.test_every_n_steps

        self.times = []

    def train_(self):

        env = self.make_env()

        time_start = time.time()

        while self.current_iteration < self.num_iterations:

            states, actions, returns, gaes, values, log_probs = self.rollout(env)

            states = torch.tensor(np.stack(states), dtype=torch.float32)
            actions = torch.stack(actions).to(torch.int64)
            returns = torch.tensor(returns, dtype=torch.float32)
            gaes = torch.tensor(gaes, dtype=torch.float32)
            log_probs = torch.stack(log_probs)
            values = torch.tensor(values, dtype=torch.float32)

            if self.normalize_advantage:
                gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

            policy_loss, value_loss, kl_divergence = self.update_networks(states, actions, log_probs, gaes, values, returns)

            self.losses.append({"policy_loss": policy_loss, "value_loss": value_loss})

            if self.current_iteration >= self.next_test:
                self.next_test += self.test_every_n_steps
                mean, std = self.test(self.num_test_episodes)
                self.test_mean_rewards.append(mean)
                self.test_std_rewards.append(std)
                runtime = time.time() - time_start
                whole_runtime = runtime / self.current_iteration * self.num_iterations
                runtime = str(timedelta(seconds=int(runtime)))
                whole_runtime = str(timedelta(seconds=int(whole_runtime)))

                description = f"Iter: [{self.current_iteration}/{self.num_iterations}] | Episode: {self.current_episode} |"
                description += f", Reward: {mean:.2f} (+/- {std:.2f})"
                description += f", KL: {kl_divergence:.2e}"
                if self.distribution == Normal:
                    description += f", std: {self.current_agent.std.detach().item():.2e}"
                description += f", Runtime: [{runtime}s/{whole_runtime}s]"
                print(description)

                self.save(self.models_folder + f"/iter_{self.current_iteration}.pkl")


    def rollout(self, env):

        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        next_values = []
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
            next_value = self.current_agent.value(torch.tensor(next_state)).detach().item()

            state = next_state
            iters += 1

            episode_length += 1
            episode_reward += reward

            if episode_length >= self.max_episode_length and self.max_episode_length != -1:
                done = True

            if episode_reward >= self.max_total_reward and self.max_total_reward != -1:
                done = True

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            values.append(value)
            next_values.append(next_value)
            log_probs.append(log_prob)

            if done:
                state, _ = env.reset()
                done = False
                self.current_episode += 1
                episode_length = 0
                episode_reward = 0

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
            gaes[slice_] = self.compute_gaes_one_episode(rewards[slice_], values[slice_], next_values[slice_])

        returns  = gaes + values

        return states, actions, returns, gaes, values, log_probs
    
    def compute_gaes_one_episode(self, rewards, values, next_values):

        next_values = np.concatenate([values[1:], [0]])
        deltas = [reward + self.discount * next_val - val for reward, val, next_val in zip(rewards, values, next_values)]

        gaes = [deltas[-1]]
        for i in reversed(range(len(deltas)-1)):
            gaes.append(deltas[i] + self.discount * self.gae_lambda * gaes[-1])

        gaes = gaes[::-1]
            
        return gaes
    
    def update_networks(self, states, actions, log_probs, gaes, values, returns):

        # Make batches of size `batch_size`
        permuted_indices = np.random.permutation(len(states))
        states = states[permuted_indices]
        actions = actions[permuted_indices]
        log_probs = log_probs[permuted_indices]
        gaes = gaes[permuted_indices].unsqueeze(-1)
        values = values[permuted_indices]
        returns = returns[permuted_indices]

        batch_indexes = np.arange(len(states))
        num_batches = len(states) // self.batch_size
        if len(states) % self.batch_size != 0:
            num_batches += 1

        for _ in range(self.n_updates):

            np.random.shuffle(batch_indexes)
            log_ratio = torch.zeros_like(log_probs)

            for i in range(num_batches):

                batch_idx = batch_indexes[i*self.batch_size:(i+1)*self.batch_size]

                self.optimizer.zero_grad()

                new_dist = self.current_agent.get_distribution(states[batch_idx])
                new_log_probs = new_dist.log_prob(actions[batch_idx])

                ratio = torch.exp(new_log_probs - log_probs[batch_idx])
                clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                clipped_loss = torch.min(ratio * gaes[batch_idx], clipped_ratio * gaes[batch_idx])

                policy_loss = torch.mean(clipped_loss)

                #VALUE_LOSS_CLIPPING = 0.2  # TODO: Make this a parameter
                #pred_values = self.current_agent.value(states[batch_idx])
                #clipped_value_loss = values + torch.clamp(pred_values - values, -VALUE_LOSS_CLIPPING, VALUE_LOSS_CLIPPING)
                #clipped_value_loss = (clipped_value_loss - returns) ** 2
                #non_clipped_value_loss = (returns - pred_values) ** 2
                #value_loss = torch.mean(torch.max(clipped_value_loss, non_clipped_value_loss))

                value_loss = torch.mean((self.current_agent.value(states[batch_idx]) - returns) ** 2)

                loss = -policy_loss + value_loss * self.value_coef

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.current_agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio[batch_idx] += log_probs[batch_idx] - new_log_probs
            
            kl_divergence = torch.mean(log_ratio).abs()
            if self.target_kl is not None:
                if kl_divergence > 1.5 * self.target_kl:
                    break

        return policy_loss.item(), value_loss.item(), kl_divergence
    
    def save(self, path):

        ppo_kwargs = {
            "env_kwargs": self.env_kwargs,
            "policy_kwargs": self.policy_kwargs,
            "value_kwargs": self.value_kwargs,
            "discount": self.discount,
            "gae_lambda": self.gae_lambda,
            "normalize_advantage": self.normalize_advantage,
            "value_coef": self.value_coef,
            "num_iterations": self.num_iterations,
            "epsilon": self.epsilon,
            "test_every_n_steps": self.test_every_n_steps,
            "update_every_n_steps": self.update_every_n_steps,
            "learning_rate": self.learning_rate,
            "update_lr_fn": self.update_lr_fn,
            "batch_size": self.batch_size,
            "n_updates": self.n_updates,
            "max_grad_norm": self.max_grad_norm,
            "target_kl": self.target_kl,
            "max_episode_length": self.max_episode_length,
            "max_total_reward": self.max_total_reward,
            "save_folder": self.save_folder,
            "normalize_observation": self.normalize_observation
        }

        saving_folders = {
            "save_folder": self.save_folder,
            "models_folder": self.models_folder,
            "plots_folder": self.plots_folder,
            "videos_folder": self.videos_folder,
        }

        model_parameters = {
            "current_agent": self.current_agent.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        running_results = {
            "current_iteration": self.current_iteration,
            "current_episode": self.current_episode,
            "test_mean_rewards": self.test_mean_rewards,
            "test_std_rewards": self.test_std_rewards,
            "losses": self.losses
        }

        data = {
            "ppo_kwargs": ppo_kwargs,
            "saving_folders": saving_folders,
            "model_parameters": model_parameters,
            "running_results": running_results
        }

        torch.save(data, path)

    def load(self, path, verbose):

        data = torch.load(path)

        # Reinitialize the model
        self.__init__(**data["ppo_kwargs"])

        # Use the same saving folders
        for key in data["saving_folders"]:
            setattr(self, key, data["saving_folders"][key])

        # Load the model parameters
        self.load_model_parameters(data["model_parameters"])

        # Load the running results
        for key in data["running_results"]:
            setattr(self, key, data["running_results"][key])

        if verbose:
            print(f"Loaded model from {path}")
            print(f"Current iteration: {self.current_iteration}, Current episode: {self.current_episode}")
            print(f"Last Test rewards: {self.test_rewards[-1]:.2f}")
            print(f"Last Losses: Policy: {self.losses[-1]['policy_loss']:.2e}, Value: {self.losses[-1]['value_loss']:.2e}")
            print(f"Last Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"Last Epsilon: {self.epsilon:.2e}")

    def load_model_parameters(self, model_parameters):

        self.current_agent.load_state_dict(model_parameters["current_agent"])
        self.optimizer.load_state_dict(model_parameters["optimizer"])
