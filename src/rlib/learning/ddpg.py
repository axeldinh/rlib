import copy
import os
import time
from datetime import timedelta
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from gymnasium.spaces import Box
from gymnasium.wrappers import ClipAction, TransformObservation, NormalizeReward, TransformReward

from rlib.agents import get_agent
from rlib.learning.base_algorithm import BaseAlgorithm
from rlib.learning.replay_buffer import ReplayBuffer


class DDPGAgent(torch.nn.Module):
    """
    Agent used for the Deep Deterministic Policy Gradient algorithm.

    For this agent, two neural networks are used: one for the policy and one for the Q-function.
    Additionally, if `twin_q=True`, two Q-functions are used, as detailed in the `TD3 paper <https://arxiv.org/pdf/1802.09477.pdf>`_.
    """

    def __init__(self, mu, q, twin_q=False):

        super().__init__()

        self.mu = mu
        self.q = q
        self.twin_q = twin_q
        if self.twin_q:
            self.q2 = copy.deepcopy(self.q)

    def get_action(self, observation):

        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()

        action = self.mu(observation)

        return action.detach().numpy()


class DDPG(BaseAlgorithm):
    """
    Implementation of the `Deep Deterministic Policy Gradient algorithm <https://arxiv.org/abs/1509.02971>`_ with options to use the improvements from `TD3 <https://arxiv.org/abs/1802.09477>`_.

    Here, a Policy agent :math:`\mu(s)` and a Q-function agent :math:`Q(s, a)` are used.
    :math:`\mu(s)` is trained to maximize :math:`Q(s, \mu(s))` and :math:`Q(s, a)` is trained to minimize :math:`(Q(s, a) - (r + \gamma Q(s', \mu(s'))))^2`.

    Because of the nature of the problem, including the fact that :math:`\mu(s)` should be differentiable, only 
    environments with continuous action spaces are supported.

    For the TD3 improvements, two Q-functions are used, and the policy is updated less frequently.

    Example:
    
    .. code-block::

        from rlib.learning import DDPG
        import gymnasium as gym

        env_kwargs = {'id': 'BipedalWalker-v3', 'hardcore': False}
        mu_kwargs = {'hidden_sizes': [256, 256]}
        q_kwargs = {'hidden_sizes': [256, 256]}

        model = DDPG(
            env_kwargs, mu_kwargs, q_kwargs,
            max_episode_length=1600, max_total_reward=-1,
            save_folder="ddpg", q_lr=3e-4, mu_lr=3e-4,
            action_noise=0.1, target_noise=0.2, delay_policy_update=2,
            twin_q=True, discount=0.99, num_episodes=2_000,
            learning_starts=0, target_update_tau=0.005,
        )

        model.train()
        model.test()
    """

    def __init__(
            self, env_kwargs, mu_kwargs, q_kwargs,
            max_episode_length=-1,
            max_total_reward=-1,
            save_folder="ddpg",
            q_lr=3e-4,
            mu_lr=3e-4,
            lr_annealing=True,
            action_noise=0.1,  # Noise added during population of the replay buffer
            target_noise=0.2,  # Noise added to target actions
            num_updates_per_iter=10,
            delay_policy_update=2,
            twin_q=True,
            discount=0.99,
            num_episodes=1000,
            learning_starts=50_000,  # Number of random samples in the replay buffer before training
            target_update_tau=0.01,  # Percentage of weights to copy from the main model to the target model
            verbose=True,
            test_every=10,
            num_test_episodes=10,
            batch_size=64,
            size_replay_buffer=100_000,
            max_grad_norm=10,
            normalize_observation=False,
            use_norm_wrappers=True,
            seed=42
            ):
        """
        Initializes the DDPG algorithm.

        :param env_kwargs: The kwargs for calling `gym.make(**env_kwargs, render_mode=render_mode)`.
        :type env_kwargs: dict
        :param mu_kwargs: The kwargs for the policy agent.
        :type mu_kwargs: dict
        :param q_kwargs: The kwargs for the Q-function agent.
        :type q_kwargs: dict
        :param max_episode_length: The maximum length of an episode, by default -1 (no limit).
        :type max_episode_length: int, optional
        :param max_total_reward: The maximum total reward to get in the episode, by default -1 (no limit).
        :type max_total_reward: float, optional
        :param save_folder: The folder where to save the models, plots and videos, by default "ddpg".
        :type save_folder: str, optional
        :param q_lr: The learning rate for the Q-function agent, by default 3e-4.
        :type q_lr: float, optional
        :param mu_lr: The learning rate for the policy agent, by default 3e-4.
        :type mu_lr: float, optional
        :param action_noise: The noise added during population of the replay buffer, by default 0.1.
        :type action_noise: float, optional
        :param target_noise: The noise added to target actions, by default 0.2.
        :type target_noise: float, optional
        :param num_updates_per_iter: The number of updates per iteration, by default 10.
        :type num_updates_per_iter: int, optional
        :param delay_policy_update: The number of Q-function updates before updating the policy, by default 2.
        :type delay_policy_update: int, optional
        :param twin_q: Whether to use two Q-functions, by default True.
        :type twin_q: bool, optional
        :param discount: The discount factor, by default 0.99.
        :type discount: float, optional
        :param num_episodes: The number of episodes to train, by default 1000.
        :type num_episodes: int, optional
        :param learning_starts: The number of random samples in the replay buffer before training, by default 50_000.
        :type learning_starts: int, optional
        :param target_update_tau: The percentage of weights to copy from the main model to the target model, by default 0.01.
        :type target_update_tau: float, optional
        :param verbose: Whether to print the progress, by default True.
        :type verbose: bool, optional
        :param test_every: The number of episodes between each test, by default 50_000.
        :type test_every: int, optional
        :param num_test_episodes: The number of episodes to test, by default 10.
        :type num_test_episodes: int, optional
        :param batch_size: The batch size for training, by default 64.
        :type batch_size: int, optional
        :param size_replay_buffer: The size of the replay buffer, by default 100_000.
        :type size_replay_buffer: int, optional
        :param max_grad_norm: The maximum norm of the gradients, by default 10.
        :type max_grad_norm: int, optional
        :param normalize_observation: Whether to normalize the observations, by default False.
        :type normalize_observation: bool, optional
        :param use_norm_wrappers: Whether to use the ClipAction, NormalizeReward and clip the observations and rewards to `[-10, 10]`. This is useful for the MuJoCo environments, by default True.
        :type use_norm_wrappers: bool, optional
        :param seed: The seed for the random number generator, by default 42.
        :type seed: int, optional
        :raises ValueError: If the action space is not continuous.
        :raises NotImplementedError: If the observation space is not 1D, 2D or 3D.

        """

        self.kwargs = locals()
        self.kwargs.pop("self")
        self.kwargs.pop("__class__")

        if use_norm_wrappers:
            envs_wrappers = [
                ClipAction, lambda env: TransformObservation(env, lambda obs: np.clip(obs, -10, 10)),
                NormalizeReward, lambda env: TransformReward(env, lambda rew: np.clip(rew, -10, 10))
            ]
        else:
            envs_wrappers = None
                        
        super().__init__(env_kwargs=env_kwargs, num_envs=1, 
                         max_episode_length=max_episode_length, max_total_reward=max_total_reward, 
                         save_folder=save_folder, normalize_observation=normalize_observation, seed=seed,
                         envs_wrappers=envs_wrappers)

        self.mu_kwargs = mu_kwargs
        self.q_kwargs = q_kwargs
        self.q_lr = q_lr
        self.mu_lr = mu_lr
        self.lr_annealing = lr_annealing
        self.discount = discount
        self.action_noise = action_noise
        self.target_noise = target_noise
        self.num_updates_per_iter = num_updates_per_iter
        self.delay_policy_update = delay_policy_update
        self.twin_q = twin_q
        self.num_episodes = num_episodes
        self.learning_starts = max(learning_starts, batch_size)
        self.target_update_tau = target_update_tau
        self.verbose = verbose
        self.test_every = test_every
        self.num_test_episodes = num_test_episodes
        self.batch_size = batch_size
        self.size_replay_buffer = size_replay_buffer
        self.max_grad_norm = max_grad_norm
        self.normalize_observation = normalize_observation

        self.current_episode = 0
        self.global_step = 0
        self.running_average = []

        if not isinstance(self.action_space, Box):
            raise ValueError("Only continuous action spaces are supported for DDPG.")
        
        mu = get_agent(self.obs_space, self.action_space, mu_kwargs)
        q_kwargs_copy = q_kwargs.copy()
        q = get_agent(self.obs_space, self.action_space, q_kwargs_copy, ddpg_q_agent=True)
        
        self.current_agent = DDPGAgent(mu, q, twin_q)
        self.target_agent = DDPGAgent(copy.deepcopy(mu), copy.deepcopy(q), twin_q)

        if self.twin_q:
            self.current_agent.q2 = copy.deepcopy(self.current_agent.q)
            self.target_agent.q2 = copy.deepcopy(self.target_agent.q)

        # Copy the parameters of the main model to the target model
        self._update_target_weights(tau=1)
        
        self.losses = []
        self.train_rewards = []
        self.mean_test_rewards = []
        self.std_test_rewards = []
        self.episodes_lengths = []
        
        # Store s_t, a_t, r_t, s_t+1, done
        self.replay_buffer = ReplayBuffer(self.size_replay_buffer)

        self.q_optimizer = torch.optim.Adam(self.current_agent.q.parameters(), lr=self.q_lr)
        self.mu_optimizer = torch.optim.Adam(self.current_agent.mu.parameters(), lr=self.mu_lr, maximize=True)

        if self.twin_q:
            self.q2_optimizer = torch.optim.Adam(self.current_agent.q2.parameters(), lr=self.q_lr)

        self.best_iteration = 0
        self.best_test_reward = -np.inf
        
    def train_(self):

        writer = SummaryWriter(os.path.join(self.save_folder, "logs"))

        env = self.make_env()

        self._populate_replay_buffer(env)

        init_episode = self.current_episode + 1

        pbar = range(init_episode, self.num_episodes + 1)

        times = []

        for _ in pbar:

            start = time.time()
            
            # Check if we need to make tests
            test_progress = (self.current_episode+1) % self.test_every == 0
            test_progress += (self.current_episode+1) == self.num_episodes

            # Run the episode

            state, _ = env.reset()
            done = False
            length_episode = 0
            episode_reward = 0

            while not done:

                action = self.current_agent.get_action(state)
                action += np.random.randn(*action.shape) * self.action_noise
                action = np.clip(action,
                                 self.action_space.low, 
                                 self.action_space.high)

                new_state, reward, done, _, _ = env.step(action)

                episode_reward += reward

                if episode_reward >= self.max_total_reward and self.max_total_reward != -1:
                    done = True

                self.replay_buffer.store(state.copy(), action.copy(), reward, new_state.copy(), done)

                state = new_state
                
                length_episode += 1

                if length_episode >= self.max_episode_length and self.max_episode_length != -1:
                    done = True

            self.global_step += length_episode

            episode_losses = {'q': [], 'mu': []}
            #for _ in range(length_episode):
            for _ in range(self.num_updates_per_iter):
                # When the episode is done, we update the weights
                loss = self.update_weights()
                episode_losses['q'].append(loss['q'])
                episode_losses['mu'].append(loss['mu'])

            if self.lr_annealing:

                self.new_mu_lr = self.mu_lr * (1 - self.current_episode / (self.num_episodes-1))
                self.new_q_lr = self.q_lr * (1 - self.current_episode / (self.num_episodes-1)) 
                
                self.mu_optimizer.param_groups[0]['lr'] = self.new_mu_lr
                self.q_optimizer.param_groups[0]['lr'] = self.new_q_lr

            loss = {'q': np.mean(episode_losses['q']), 'mu': np.mean(episode_losses['mu'])}
            self.losses.append(loss)

            writer.add_scalar("Losses/Q", loss['q'], self.global_step)
            writer.add_scalar("Losses/Mu", loss['mu'], self.global_step)
            
            # Save the rewards
            self.episodes_lengths.append(length_episode)
            self.train_rewards.append(episode_reward)

            writer.add_scalar("Train/Reward", episode_reward, self.global_step)
            writer.add_scalar("Train/Episode Length", length_episode, self.global_step)

            self.current_episode += 1

            if test_progress:
                mean, std = self.test(self.num_test_episodes)
                self.mean_test_rewards.append(mean)
                self.std_test_rewards.append(std)

                writer.add_scalar("Test/Mean Reward", mean, self.global_step)
                writer.add_scalar("Test/Std Reward", std, self.global_step)

                if self.running_average.__len__() == 0:
                    self.running_average.append(mean)
                else:
                    self.running_average.append(0.9 * self.running_average[-1] + 0.1 * mean)

                writer.add_scalar("Train/Running Reward", self.running_average[-1], self.global_step)

                if mean > self.best_test_reward:
                    self.best_test_reward = mean
                    self.best_iteration = self.current_episode
                    self.save(self.models_folder + "/best.pt")
                
                self.save(self.models_folder + f"/iter_{self.current_episode}.pt")

            run_time = time.time() - start
            times.append(run_time)

            if self.verbose and test_progress:
                description = f"Episode [{self.current_episode}/{self.num_episodes}]"
                description += ", Test Reward: {:.2f} (+/- {:.2f})".format(self.mean_test_rewards[-1], self.std_test_rewards[-1])
                description += ", Running Average: {:.2f}".format(self.running_average[-1])
                total_time = int(np.mean(times) * self.num_episodes)
                total_time = str(timedelta(seconds=int(total_time)))
                current_time = int(np.sum(times))
                current_time = str(timedelta(seconds=int(current_time)))
                description += ", Time: [{}s/{}s]".format(current_time, total_time)
                print(description)

        env.close()
        writer.close()
                
    def _update_target_weights(self, tau=0.01):
        """
        Updates the target weights using the current weights.

        It uses the formula:

        .. math::

            \\theta_{target} = \\tau \\theta_{current} + (1 - \\tau) \\theta_{target}

        """
        # Update Q Network weights 
        for target_param, param in zip(self.target_agent.q.parameters(), self.current_agent.q.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        if self.twin_q:
            for target_param, param in zip(self.target_agent.q2.parameters(), self.current_agent.q2.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        if (self.current_episode + 1) % self.delay_policy_update != 0:
            return

        # Update Mu Network weights
        for target_param, param in zip(self.target_agent.mu.parameters(), self.current_agent.mu.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def _populate_replay_buffer(self, env):
        """
        Plays random actions in the environment to populate the replay buffer, until the number of samples is equal to `learning_starts`.

        :param env: The environment to use.
        :type env: gymnasium.ENV
        """

        obs, _ = env.reset()
        done = False

        while self.replay_buffer.size < self.learning_starts:

            if self.replay_buffer.size >= self.size_replay_buffer:
                break

            action = env.action_space.sample()
            new_obs, reward, done, _, _ = env.step(action)
            self.replay_buffer.store(obs.copy(), action.copy(), reward, new_obs.copy(), done)
            obs = new_obs

            if done:
                obs, _ = env.reset()
                done = False

        if self.verbose:
            print(f"Replay buffer populated with {self.replay_buffer.size} samples.")

    def update_weights(self):
        """
        Updates the neural networks using the replay buffer.
        """

        s_t, a_t, r_t, s_t_1, done_ = self.replay_buffer.sample(self.batch_size)

        s_t = torch.tensor(s_t)
        a_t = torch.tensor(a_t)
        r_t = torch.tensor(r_t).squeeze()
        s_t_1 = torch.tensor(s_t_1)
        done_ = torch.tensor(done_).squeeze()

        # Update the Q function
        target_a_t = self.target_agent.mu(s_t_1)
        target_a_t += torch.randn_like(target_a_t) * self.target_noise
        target_a_t = torch.clamp(target_a_t,
                                 torch.tensor(self.action_space.low).float(), 
                                 torch.tensor(self.action_space.high).float())
        target_q_t = self.target_agent.q(torch.cat([s_t_1, target_a_t], dim=-1)).squeeze().detach()

        if self.twin_q:
            target_q2_t = self.target_agent.q2(torch.cat([s_t_1, target_a_t], dim=-1)).squeeze().detach()
            target_q_t = torch.min(target_q_t, target_q2_t)

        target = r_t + self.discount * target_q_t * (1 - done_)
        target = target.detach()
        
        q = self.current_agent.q(torch.cat([s_t, a_t], dim=-1)).squeeze()

        loss_q = F.mse_loss(q, target)
        
        self.current_agent.q.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(self.current_agent.q.parameters(), self.max_grad_norm)
        self.q_optimizer.step()

        if self.twin_q:
            loss_q2 = F.mse_loss(self.current_agent.q2(torch.cat([s_t, a_t], dim=-1)).squeeze(), target)
            self.current_agent.q2.zero_grad()
            loss_q2.backward()
            torch.nn.utils.clip_grad_norm_(self.current_agent.q2.parameters(), self.max_grad_norm)
            self.q2_optimizer.step()

        # Update the Policy function
        if (self.current_episode + 1) % self.delay_policy_update == 0:
            actions = self.current_agent.mu(s_t)
            loss_mu = self.current_agent.q(torch.cat([s_t, actions], dim=-1)).mean()

            self.current_agent.q.zero_grad()
            if self.twin_q:
                self.current_agent.q2.zero_grad()
            self.current_agent.mu.zero_grad()
            loss_mu.backward()        
            torch.nn.utils.clip_grad_norm_(self.current_agent.mu.parameters(), self.max_grad_norm)
            self.mu_optimizer.step()

            # Update the target weights    
            self._update_target_weights(tau=self.target_update_tau)

        else:
            if self.losses.__len__() == 0:
                loss_mu = torch.tensor(0)
            else:
                loss_mu = torch.tensor(self.losses[-1]['mu'])

        return {'q': loss_q.item(), 'mu': loss_mu.item()}
    
    def save(self, path):

        kwargs = self.kwargs.copy()

        saving_folders = {
            "save_folder": self.save_folder,
            "models_folder": self.models_folder,
            "plots_folder": self.plots_folder,
            "videos_folder": self.videos_folder,
        }

        model_parameters = {
            "current_agent": self.current_agent.state_dict(),
            "target_agent": self.target_agent.state_dict(),
            "mu_optimizer": self.mu_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
        }

        running_results = {
            "current_episode": self.current_episode,
            "losses": self.losses,
            "train_rewards": self.train_rewards,
            "mean_test_rewards": self.mean_test_rewards,
            "std_test_rewards": self.std_test_rewards,
            "episodes_lengths": self.episodes_lengths,
            "running_average": self.running_average,
            "best_iteration": self.best_iteration,
            "best_test_reward": self.best_test_reward,
        }

        data = {
            "kwargs": kwargs,
            "saving_folders": saving_folders,
            "model_parameters": model_parameters,
            "running_results": running_results,
        }

        torch.save(data, path)

    def load(self, path, verbose=True):

        data = torch.load(path)

        # Reinitialize the model using the kwargs
        self.__init__(**data['kwargs'])

        # The exact saving folders need to fixed again (their is an increment during init)
        for key in data['saving_folders'].keys():
            setattr(self, key, data['saving_folders'][key])

        # Load the models and optimizers parameters
        self.load_model_parameters(data)

        # Load the running results
        for key in data['running_results'].keys():
            setattr(self, key, data['running_results'][key])

        if verbose:
            print(f"Loaded model from {path}")
            print(f"Current Episode: [{self.current_episode}/{self.num_episodes}]")
            print(f"Maximum Episode Length: {self.max_episode_length}, Maximum Total Reward: {self.max_total_reward}")
            print(f"Q Learning Rate: {self.q_lr}, Mu Learning Rate: {self.mu_lr}")
            print(f"Discount: {self.discount}, Target Update Tau: {self.target_update_tau}")
            print(f"Action Noise: {self.action_noise}, Target Noise: {self.target_noise}")
            print(f"Delay Policy Update: {self.delay_policy_update}, Twin Q: {self.twin_q}")
            print(f"Size Replay Buffer: {self.replay_buffer.size}, Max Size: {self.size_replay_buffer}, Learning Starts: {self.learning_starts}")
            print(f"Normalize Observation: {self.normalize_observation}, Verbose: {self.verbose}")
            print(f"Test Every: {self.test_every}, Number of Test Episodes: {self.num_test_episodes}")
            print(f"Batch Size: {self.batch_size}, Max Grad Norm: {self.max_grad_norm}")

            print("Running Results:")
            print(f"\tMu Loss: {self.losses[-1]['mu']}")
            print(f"\tQ Loss: {self.losses[-1]['q']}")
            print(f"\tRunning Reward: {self.running_average[-1]}")

    def save_plots(self):

        import matplotlib.pyplot as plt

        x_range = np.arange(self.test_every, self.test_every * len(self.mean_test_rewards) + 1, self.test_every)
        if len(x_range) != len(self.mean_test_rewards):
            x_range = np.append(x_range, self.num_time_steps)
        plt.plot(x_range, self.mean_test_rewards, c="red", label="Test Reward")
        plt.fill_between(
            x_range,
            np.array(self.mean_test_rewards) - np.array(self.std_test_rewards),
            np.array(self.mean_test_rewards) + np.array(self.std_test_rewards),
            alpha=0.5, color="red"
        )
        plt.plot(x_range, self.running_average, c="blue", label="Running Average")
        plt.xlabel("Number of Episodes")
        plt.ylabel("Reward")
        plt.legend()
        plt.savefig(self.plots_folder + "/mean_test_rewards.png", bbox_inches="tight")
        plt.close()

        plt.plot(range(1, len(self.episodes_lengths)+1), self.episodes_lengths)
        plt.xlabel("Number of episodes")
        plt.ylabel("Episode length")
        plt.savefig(self.plots_folder + "/episode_lengths.png", bbox_inches="tight")
        plt.close()

        plt.plot(range(1, len(self.train_rewards)+1), self.train_rewards)
        plt.xlabel("Number of episodes")
        plt.ylabel("Reward")
        plt.savefig(self.plots_folder + "/train_rewards.png", bbox_inches="tight")
        plt.close()

        losses_mu = [loss['mu'] for loss in self.losses]
        plt.plot(range(1, len(self.losses) + 1), losses_mu)
        plt.xlabel("Number of Episodes")
        plt.ylabel("Policy Loss (Higher is better)")
        plt.savefig(self.plots_folder + "/policy_losses.png", bbox_inches="tight")
        plt.close()

        losses_q = [loss['q'] for loss in self.losses]
        plt.plot(range(1, len(losses_q) + 1), losses_q)
        plt.xlabel("Number of Episodes")
        plt.ylabel("Q Loss (Lower is better)")
        plt.yscale("log")
        plt.savefig(self.plots_folder + "/q_losses.png", bbox_inches="tight")
        plt.close()

        print("Figures saved in ", self.plots_folder)

    def load_from_path(path):

        data = torch.load(path)

        model = DDPG(**data['ddpg_kwargs'])
        model.load_model_parameters(data)
        
        for key in data['running_results'].keys():
            setattr(model, key, data['running_results'][key])

        return model
    
    def load_model_parameters(self, data):

        self.current_agent.load_state_dict(data['model_parameters']['current_agent'])
        self.target_agent.load_state_dict(data['model_parameters']['target_agent'])
        self.mu_optimizer.load_state_dict(data['model_parameters']['mu_optimizer'])
        self.q_optimizer.load_state_dict(data['model_parameters']['q_optimizer'])
