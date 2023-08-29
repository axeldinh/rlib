import datetime
import time
import os

import numpy as np
import torch
import torch.nn .functional as F
from torch.utils.tensorboard import SummaryWriter

from gymnasium.spaces import Discrete

from rlib.learning.base_algorithm import BaseAlgorithm
from rlib.agents import get_agent
from rlib.learning.replay_buffer import ReplayBuffer

class DeepQLearningAgent(torch.nn.Module):

    def __init__(self, obs_space, action_space, agent_kwargs):

        super().__init__()

        self.agent = get_agent(obs_space, action_space, agent_kwargs)

    def forward(self, x):
        return self.agent(x)
    
    def get_action(self, x):

        is_numpy = isinstance(x, np.ndarray)
        if is_numpy:
            x = torch.tensor(x, dtype=torch.float32)
        logits = self.forward(x)
        action = torch.argmax(logits, dim=-1)
        if is_numpy:
            action = action.detach().numpy()
        return action


class DeepQLearning(BaseAlgorithm):
    """
    Deep Q-Learning algorithm.

    The Q-Table is replaced by a neural network that approximates the Q-Table.

    The neural network is updated using the following formula:

    .. math::

        Q(s_t, a_t) = Q(s_t, a_t) + \\alpha \\left(r_{t+1} + \\gamma \\max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \\right)

    where :math:`\\alpha` is the learning rate and :math:`\\gamma` the discount factor.

    Hence, the method is only suitable for discrete action spaces.

    An epsilon greedy policy is used to select the actions, and the actions are stored in a :class:`ReplayBuffer` 
    before being used to update the neural network.

    Example:

    .. code-block::

        import gymnasium as gym
        from rlib.learning import DeepQLearning
        
        env_kwargs = {'id': 'CartPole-v1'}
        agent_kwargs = {'hidden_sizes': [10]}

        model = DeepQLearning(
            env_kwargs, agent_kwargs,
            lr=0.03, discount=0.99,
            epsilon_greedy=0.1, epsilon_decay=0.9999, epsilon_min=0.01
            )

        model.train()
        model.test()
        model.save_plots()

    """

    def __init__(
            self, env_kwargs, agent_kwargs,
            max_episode_length=-1,
            max_total_reward=-1,
            save_folder="deep_qlearning",
            lr=3e-4,
            discount=0.99,
            epsilon_greedy=0.1,
            epsilon_decay=0.99,
            epsilon_min=0.01,
            num_time_steps=100_000,
            learning_starts=50_000,
            update_every=4,
            main_target_update=100,
            verbose=True,
            test_every=50_000,
            num_test_episodes=10,
            batch_size=64,
            size_replay_buffer=100_000,
            max_grad_norm=10,
            normalize_observation=False,
            seed=42
            ):
        """
        Initializes the DeepQLearning algorithm.

        :param env_kwargs: The kwargs for calling `gym.make(**env_kwargs, render_mode=render_mode)`.
        :type env_kwargs: dict
        :param agent_kwargs: The kwargs for calling `get_agent(obs_space, action_space, **agent_kwargs)`.
        :type agent_kwargs: dict
        :param max_episode_length: The maximum length of an episode, by default -1 (no limit).
        :type max_episode_length: int, optional
        :param max_total_reward: The maximum total reward to get in the episode, by default -1 (no limit).
        :type max_total_reward: float, optional
        :param save_folder: The folder where to save the model, by default "deep_qlearning".
        :type save_folder: str, optional
        :param lr: The learning rate, by default 3e-4.
        :type lr: float, optional
        :param discount: The discount factor, by default 0.99.
        :type discount: float, optional
        :param epsilon_greedy: The probability to take a random action, by default 0.1.
        :type epsilon_greedy: float, optional
        :param epsilon_decay: The decay of the epsilon greedy, by default 0.99.
        :type epsilon_decay: float, optional
        :param epsilon_min: The minimum value of epsilon greedy, by default 0.01.
        :type epsilon_min: float, optional
        :param num_time_steps: The number of time steps to train the agent, by default 100_000.
        :type num_time_steps: int, optional
        :param learning_starts: The number of time steps before starting to train the agent, by default 50_000.
        :type learning_starts: int, optional
        :param update_every: The number of time steps between each update of the neural network, by default 4.
        :type update_every: int, optional
        :param main_target_update: The number of time steps between each update of the target network, by default 100.
        :type main_target_update: int, optional
        :param verbose: Whether to print the results, by default True.
        :type verbose: bool, optional
        :param test_every: The number of time steps between each test, by default 50_000.
        :type test_every: int, optional
        :param num_test_episodes: The number of episodes to test the agent, by default 10.
        :type num_test_episodes: int, optional
        :param batch_size: The batch size, by default 64.
        :type batch_size: int, optional
        :param size_replay_buffer: The size of the replay buffer, by default 100_000.
        :type size_replay_buffer: int, optional
        :param max_grad_norm: The maximum norm of the gradients, by default 10.
        :type max_grad_norm: int, optional
        :param normalize_observation: Whether to normalize the observation in `[-1, 1]`, by default False.
        :type normalize_observation: bool, optional
        :param seed: The seed for the environment, by default 42.
        :type seed: int, optional

        """

        self.kwargs = locals()
        self.kwargs.pop("self")
        self.kwargs.pop("__class__")
        
        
        super().__init__(env_kwargs=env_kwargs, num_envs=1,
                         max_episode_length=max_episode_length, max_total_reward=max_total_reward, 
                         save_folder=save_folder, normalize_observation=normalize_observation, seed=seed)

        self.agent_kwargs = agent_kwargs
        self.lr = lr
        self.discount = discount
        self.epsilon_greedy = epsilon_greedy
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_time_steps = num_time_steps
        self.learning_starts = max(learning_starts, batch_size)
        self.update_every = update_every
        self.main_target_update = main_target_update
        self.verbose = verbose
        self.test_every = test_every
        self.num_test_episodes = num_test_episodes
        self.batch_size = batch_size
        self.size_replay_buffer = size_replay_buffer
        self.max_grad_norm = max_grad_norm

        self.current_time_step = 0
        self.running_average = []

        if not isinstance(self.action_space, Discrete):
            raise ValueError("The action space must be discrete. Current action space: {}".format(self.action_space))
        
        self.current_agent = DeepQLearningAgent(self.obs_space, self.action_space, self.agent_kwargs)
        self.target_agent = DeepQLearningAgent(self.obs_space, self.action_space, self.agent_kwargs)

        # Copy the parameters of the main model to the target model
        for target_param, param in zip(self.target_agent.parameters(), self.current_agent.parameters()):
            target_param.data.copy_(param.data)
        
        self.losses = []
        self.train_rewards = []
        self.mean_test_rewards = []
        self.std_test_rewards = []
        self.episodes_lengths = []
        
        # Store s_t, a_t, r_t, s_t+1, done
        self.replay_buffer = ReplayBuffer(max_size=self.size_replay_buffer)

        self.optimizer = torch.optim.Adam(self.current_agent.parameters(), lr=self.lr)

        self.best_iteration = 0
        self.best_test_reward = -np.inf
        
    def train_(self):

        writer = SummaryWriter(os.path.join(self.save_folder, "logs"))
        
        env = self.make_env().envs[0]

        self._populate_replay_buffer(env)  # Populate the replay buffer with random samples

        pbar = range(self.num_time_steps)

        state, _ = env.reset()
        done = False
        length_episode = 0
        episode_reward = 0

        times = []
        time_start = time.time()

        for n in pbar:

            test_progress = (n+1) % self.test_every == 0
            test_progress += (n+1) == self.num_time_steps
                
            if np.random.rand() < self.epsilon_greedy:
                action = env.action_space.sample()
            else:
                action = self.target_agent.get_action(state)

            new_state, reward, done, _, _ = env.step(action)

            episode_reward += reward

            if episode_reward >= self.max_total_reward and self.max_total_reward != -1:
                done = True

            self.replay_buffer.store(state.copy(), action, reward, new_state.copy(), done)

            if self.current_time_step % self.update_every == 0:
                loss = self.update_weights()
                self.losses.append(loss.item())
                writer.add_scalar("Loss", loss.item(), self.current_time_step)
                if self.current_time_step % self.main_target_update == 0:
                    for target_param, param in zip(self.target_agent.parameters(), self.current_agent.parameters()):
                        target_param.data.copy_(param.data)

            state = new_state
            
            # Update epsilon greedy, to take less random actions
            self.epsilon_greedy = max(self.epsilon_min, self.epsilon_greedy * self.epsilon_decay)

            length_episode += 1

            if length_episode >= self.max_episode_length and self.max_episode_length != -1:
                done = True

            self.current_time_step += 1

            if test_progress:
                mean, std = self.test(self.num_test_episodes)
                self.mean_test_rewards.append(mean)
                self.std_test_rewards.append(std)
                writer.add_scalar("Test/Mean Reward", mean, self.current_time_step)
                writer.add_scalar("Test/Std Reward", std, self.current_time_step)

                if self.running_average.__len__() == 0:
                    self.running_average.append(mean)
                else:
                    self.running_average.append(0.9 * self.running_average[-1] + 0.1 * mean)

                writer.add_scalar("Test/Running Average", self.running_average[-1], self.current_time_step)

                if mean > self.best_test_reward:
                    self.best_test_reward = mean
                    self.best_iteration = self.current_time_step
                    self.save(self.models_folder + "/best.pkl")

                self.save(self.models_folder + f"/iter_{self.current_time_step}.pkl")

            if self.verbose and test_progress:
                description = f"TimeStep: [{self.current_time_step}/{self.num_time_steps}]"
                description += ", Test Reward: {:.2f} (+/- {:.2f})".format(self.mean_test_rewards[-1], self.std_test_rewards[-1])
                description += ", Running Average: {:.2f}".format(self.running_average[-1])
                times.append(time.time() - time_start)
                total_time = np.mean(times) * self.num_time_steps / self.test_every
                current_time = np.sum(times)
                total_time = str(datetime.timedelta(seconds=int(total_time)))
                current_time = str(datetime.timedelta(seconds=int(current_time)))
                description += f", Time: [{self.current_time_step}/{self.num_time_steps}]"
                print(description)
                time_start = time.time()

            if done:
                state, _ = env.reset()
                done = False
                self.episodes_lengths.append(length_episode)
                self.train_rewards.append(episode_reward)
                writer.add_scalar("Train/Reward", episode_reward, self.current_time_step)
                writer.add_scalar("Train/Episode Length", length_episode, self.current_time_step)
                length_episode = 0
                episode_reward = 0

    def _populate_replay_buffer(self, env):
        """
        Populate the replay buffer with random samples from the environment.

        This is done until the replay buffer is filled with :attr:`learning_starts` samples.
        Furthermore, the actions are sampled randomly with probability :attr:`epsilon_greedy`.

        :param env: The environment to sample from.
        :type env: gymnasium.ENV
        """

        obs, _ = env.reset()
        done = False

        while len(self.replay_buffer) < self.learning_starts:

            if len(self.replay_buffer) >= self.size_replay_buffer:
                break

            if np.random.rand() < self.epsilon_greedy:
                action = env.action_space.sample()
            else:
                action = self.target_agent.get_action(obs)
            new_obs, reward, done, _, _ = env.step(action)
            self.replay_buffer.store(obs.copy(), action, reward, new_obs.copy(), done)
            obs = new_obs

            if done:
                obs, _ = env.reset()
                done = False

        if self.verbose:
            print(f"Replay buffer populated with {len(self.replay_buffer)} samples.")

    def update_weights(self):
        """
        Update the weights of the neural network.
        
        From the ::attr:`replay_buffer`, a batch of size :attr:`batch_size` is used to update the weights of the neural network using the following loss:

        """
        
        s_t, a_t, r_t, s_t_1, done_ = self.replay_buffer.sample(self.batch_size)
        
        s_t = torch.tensor(s_t)
        a_t = torch.tensor(a_t).type(torch.int64)
        r_t = torch.tensor(r_t).squeeze()
        s_t_1 = torch.tensor(s_t_1)
        done_ = torch.tensor(done_).squeeze()

        q = self.current_agent(s_t)
        q = q.gather(1, a_t).squeeze(1)
        next_q = torch.amax(self.target_agent(s_t_1).detach(), dim=-1)
        target = r_t + self.discount * next_q * (1 - done_)

        loss = F.smooth_l1_loss(q, target)

        self.current_agent.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.current_agent.parameters(), self.max_grad_norm)

        self.optimizer.step()

        return loss
    
    def save(self, path):

        kwargs = self.kwargs.copy()

        saving_folders = {
            "save_folder": self.save_folder,
            "models_folder": self.models_folder,
            "plots_folder": self.plots_folder,
            "videos_folder": self.videos_folder
        }

        model_parameters = {
            "current_agent": self.current_agent.state_dict(),
            "target_agent": self.target_agent.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        running_results = {
            "current_time_step": self.current_time_step,
            "running_average": self.running_average,
            "losses": self.losses,
            "train_rewards": self.train_rewards,
            "mean_test_rewards": self.mean_test_rewards,
            "std_test_rewards": self.std_test_rewards,
            "episodes_lengths": self.episodes_lengths,
            "replay_buffer": self.replay_buffer,
            "best_iteration": self.best_iteration,
            "best_test_reward": self.best_test_reward
        }

        data = {
            "kwargs": kwargs,
            "saving_folders": saving_folders,
            "model_parameters": model_parameters,
            "running_results": running_results
        }

        torch.save(data, path)

    def load(self, path, verbose=True):

        data = torch.load(path)

        self.__init__(**data['kwargs'])

        for key in data['saving_folders'].keys():
            setattr(self, key, data['saving_folders'][key])

        self.load_model_parameters(data)

        for key in data['running_results'].keys():
            setattr(self, key, data['running_results'][key])

        if verbose:
            print("Model loaded from: ", path)
            print(f"Current Iteration: [{self.current_time_step}/{self.num_time_steps}]")
            print(f"Learning Rate: {self.lr}, Discount: {self.discount}")
            print(f"Epsilon Greedy: {self.epsilon_greedy}, Epsilon Decay: {self.epsilon_decay}, Epsilon Min: {self.epsilon_min}")
            print(f"Learning Starts: {self.learning_starts}, Update Every: {self.update_every}, Main Target Update: {self.main_target_update}")
            print(f"Test Every: {self.test_every}, Num Test Episodes: {self.num_test_episodes}")
            print(f"Batch Size: {self.batch_size}, Size Replay Buffer: {self.size_replay_buffer}")
            print(f"Max Grad Norm: {self.max_grad_norm}, Normalize Observation: {self.normalize_observation}")
            print(f"Best Iteration: {self.best_iteration}, Best Test Reward: {self.best_test_reward}")

            print("Running Results:")
            print(f"\tRunning Average: {self.running_average[-1]}")
            print(f"\tTest Rewards: {self.mean_test_rewards[-1]} (+/- {self.std_test_rewards[-1]})")
            print(f"\tLoss: {self.losses[-1]}")

    def load_model_parameters(self, data):

        self.current_agent.load_state_dict(data['model_parameters']['current_agent'])
        self.target_agent.load_state_dict(data['model_parameters']['target_agent'])
        self.optimizer.load_state_dict(data['model_parameters']['optimizer'])
    
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
        plt.xlabel("Number of iterations")
        plt.ylabel("Reward")
        plt.legend()
        plt.savefig(self.plots_folder + "/mean_test_rewards.png", bbox_inches="tight")
        plt.close()


        x_range = [length for length in self.episodes_lengths]
        x_range = np.cumsum(x_range)
        plt.plot(x_range, self.episodes_lengths)
        plt.xlabel("Number of iterations")
        plt.ylabel("Episode length")
        plt.savefig(self.plots_folder + "/episode_lengths.png", bbox_inches="tight")
        plt.close()

        plt.plot(x_range, self.train_rewards)
        plt.xlabel("Number of iterations")
        plt.ylabel("Reward")
        plt.savefig(self.plots_folder + "/train_rewards.png", bbox_inches="tight")
        plt.close()

        plt.plot(range(1, len(self.losses) * self.batch_size + 1, self.batch_size), self.losses)
        plt.xlabel("Number of iterations")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.savefig(self.plots_folder + "/losses.png", bbox_inches="tight")
        plt.close()

        print("Figures saved in ", self.plots_folder)
