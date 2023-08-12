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

    def __init__(self, 
                 env_kwargs,                   
                 policy_kwargs,
                 value_kwargs,
                 num_envs=2,
                 discount=0.99, 
                 gae_lambda=0.95,
                 normalize_advantage=True,
                 value_coef=0.5,
                 num_iterations=100000,
                 epsilon=0.2,
                 test_every_n_steps=1000,
                 num_test_episodes=10,
                 update_every_n_steps=500,
                 learning_rate=3e-4,
                 lr_annealing=True,
                 update_lr_fn=None,
                 batch_size=64,
                 n_updates=10,
                 max_grad_norm=0.5,
                 target_kl=None,
                 max_episode_length=-1,
                 max_total_reward=-1,
                 save_folder="ppo",
                 normalize_observation=False,
                 seed=42):
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

        super().__init__(env_kwargs=env_kwargs, num_envs=num_envs, max_episode_length=max_episode_length,
                         max_total_reward=max_total_reward, save_folder=save_folder,
                         normalize_observation=normalize_observation, seed=seed)
        
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
        self.lr_annnealing = lr_annealing

        # Build the actor and critic

        if self.obs_shape.__len__() == 1:

            num_obs = self.obs_shape[0]
            num_actions = self.action_shape[0]

            policy_kwargs["input_size"] = num_obs
            policy_kwargs["output_size"] = num_actions
            policy_kwargs['requires_grad'] = True
            policy_kwargs['init_weights'] = 'ppo_actor'

            value_kwargs["input_size"] = num_obs
            value_kwargs["output_size"] = 1
            value_kwargs['requires_grad'] = True
            value_kwargs['init_weights'] = 'ppo_critic'

            self.policy = get_agent("mlp", **policy_kwargs)
            self.advantage = get_agent("mlp", **value_kwargs)

        else:
            raise NotImplementedError("PPO only supports 1D observations currently.")
        
        if self.action_space_type == "discrete":
            self.distribution = "categorical"
        elif self.action_space_type == "continuous":
            self.distribution = "normal"
        
        self.current_agent = PPOAgent(self.distribution, self.policy, self.advantage)
        
        self.optimizer = torch.optim.Adam(self.current_agent.policy_agent.parameters(), lr=self.learning_rate, eps=1e-5)

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

            if self.lr_annnealing:

                new_lr = self.learning_rate * (1 - self.current_iteration / self.num_iterations)
                self.optimizer.param_groups[0]['lr'] = new_lr

            states, actions, returns, gaes, values, log_probs = self.rollout(env)

            states = torch.tensor(np.stack(states), dtype=torch.float32).view(-1, *self.obs_shape)
            actions = torch.tensor(actions).to(torch.int64).view(-1)
            returns = torch.tensor(returns, dtype=torch.float32).view(-1)
            gaes = torch.tensor(gaes, dtype=torch.float32).view(-1)
            log_probs = torch.tensor(log_probs, dtype=torch.float32).view(-1)
            values = torch.tensor(values, dtype=torch.float32).view(-1)

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

        states = np.zeros((self.update_every_n_steps, self.num_envs, *self.obs_shape))
        if self.distribution == 'categorical':
            actions = np.zeros((self.update_every_n_steps, self.num_envs))
        elif self.distribution == 'normal':
            actions = np.zeros((self.update_every_n_steps, self.num_envs, self.action_shape[0]))
        rewards = np.zeros((self.update_every_n_steps, self.num_envs))
        dones = np.zeros((self.update_every_n_steps, self.num_envs))
        values = np.zeros((self.update_every_n_steps, self.num_envs))
        next_values = np.zeros((self.update_every_n_steps, self.num_envs))
        log_probs = np.zeros((self.update_every_n_steps, self.num_envs))

        episodes_rewards = np.array([])
        episodes_lengths = np.array([])

        state, _ = env.reset()
        done = False

        for iter_ in range(self.update_every_n_steps):
            
            state = torch.tensor(state, dtype=torch.float32)
            dist, value = self.current_agent(state)
            action = dist.sample().detach()
            log_prob = dist.log_prob(action).detach()
            value = value.detach().squeeze(-1)

            next_state, reward, done, _, infos = env.step(action.numpy())
            next_value = self.current_agent.value(torch.tensor(next_state, dtype=torch.float32)).detach().squeeze(-1)

            states[iter_] = state
            actions[iter_] = action
            rewards[iter_] = reward
            dones[iter_] = done
            values[iter_] = value
            next_values[iter_] = next_value
            log_probs[iter_] = log_prob

            if np.any(done):
                episodes_rewards = np.concatenate([episodes_rewards, infos["episode"]["r"]])
                episodes_lengths = np.concatenate([episodes_lengths, infos["episode"]["l"]])

            state = next_state

        self.current_iteration += self.update_every_n_steps * self.num_envs

        self.current_episode += len(episodes_rewards)

        # Value Bootstrap
        with torch.no_grad():

            next_value = self.current_agent.value(torch.tensor(next_state, dtype=torch.float32)).squeeze(-1).numpy()

            gaes = np.zeros_like(rewards)
            last_gae = 0

            for t in reversed(range(self.update_every_n_steps)):
                
                if t == self.update_every_n_steps - 1:
                    next_non_terminal = 1.0 - dones[t]
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - dones[t+1]
                    next_values = values[t+1]

                delta = rewards[t] + self.discount * next_values * next_non_terminal - values[t]
                gaes[t] = last_gae = delta + self.discount * self.gae_lambda * next_non_terminal * last_gae

            returns = gaes + values

        return states, actions, returns, gaes, values, log_probs

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

                VALUE_LOSS_CLIPPING = 0.2  # TODO: Make this a parameter
                pred_values = self.current_agent.value(states[batch_idx])
                clipped_value_loss = values[batch_idx] + torch.clamp(pred_values - values[batch_idx], -VALUE_LOSS_CLIPPING, VALUE_LOSS_CLIPPING)
                clipped_value_loss = (returns[batch_idx] - clipped_value_loss) ** 2
                non_clipped_value_loss = (returns[batch_idx] - pred_values) ** 2
                value_loss = torch.mean(torch.max(clipped_value_loss, non_clipped_value_loss))

                #value_loss = torch.mean((self.current_agent.value(states[batch_idx]) - returns) ** 2)

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
