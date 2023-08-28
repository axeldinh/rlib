
import os
from tqdm import trange
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from rlib.learning.base_algorithm import BaseAlgorithm
from rlib.agents import get_agent

class QLearning(BaseAlgorithm):
    """
    Applies the QLearning algorithm to the environment.

    The Q-Table is updated using the following formula:

    .. math::

        Q(s_t, a_t) = Q(s_t, a_t) + \\alpha \\left(r_{t+1} + \\gamma \\max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \\right)

    where :math:`\\alpha` is the learning rate and :math:`\\gamma` the discount factor.

    An epsilon greedy policy is used to select the actions.

    Example:

    .. code-block:: python

        import gymnasium
        from rlib.learning import QLearning
        from rlib.agents import QTable

        env_fn = lambda render_mode=None: gymnasium.make('MountainCar-V0', render_mode=render_mode)
        agent_fn = lambda: QTable(env_fn(), grid_size=20)
        model = QLearning(env_fn, agent_fn, lr=0.03, discount=0.99, epsilon_greedy=0.1, epsilon_decay=0.9999, epsilon_min=0.01))

        model.train()
        mean, std = model.test(10)

        print("Mean reward:", mean)
        print("Standard deviation:", std)

        model.save_plots()
        model.save_videos()

    """

    def __init__(
            self, env_kwargs, agent_kwargs,
            max_episode_length=-1, max_total_reward=-1,
            save_folder="qlearning", num_iterations=1000,
            lr=0.03, discount=0.99, epsilon_greedy=0.9, epsilon_decay=0.9999, epsilon_min=0.01,
            test_every=10, num_test_episodes=5, verbose=True, seed=42
            ):
        """
        Initialize the QLearning algorithm.

        :param env_kwargs: The kwargs for calling `gym.make(**env_kwargs, render_mode=render_mode)`.
        :type env_kwargs: dict
        :param agent_kwargs: Kwargs to call `rlib.agents.q_table.QTable(**agent_kwargs)`.
        :type agent_kwargs: dict
        :param max_episode_length: The maximum length of an episode, by default -1 (no limit).
        :type max_episode_length: int, optional
        :param max_total_reward: The maximum total reward to get in the episode, by default -1 (no limit).
        :type max_total_reward: float, optional
        :param save_folder: The path of the folder where to save the results. Default is "results"
        :param num_iterations: The number of episodes to train. Default is 1000.
        :type num_iterations: int, optional
        :type save_folder: str, optional
        :param lr: The learning rate. Default is 0.03.
        :type lr: float, optional
        :param discount: The discount factor. Default is 0.99.
        :type discount: float, optional
        :param epsilon_greedy: The epsilon greedy parameter. Default is 0.9.
        :type epsilon_greedy: float, optional
        :param epsilon_decay: The epsilon decay parameter. Default is 0.9999.
        :type epsilon_decay: float, optional
        :param epsilon_min: The minimum epsilon value. Default is 0.01.
        :type epsilon_min: float, optional
        :param test_every: The number of episodes between each save. Default is 10.
        :type test_every: int, optional
        :param num_test_episodes: The number of episodes to test. Default is 5.
        :type num_test_episodes: int, optional
        :param verbose: Whether to print the results of each episode. Default is True.
        :type verbose: bool, optional
        :param seed: The seed for the environment. Default is 42.
        :type seed: int, optional

        """

        self.kwargs = locals()
        self.kwargs.pop("self")
        self.kwargs.pop("__class__")

        super().__init__(env_kwargs=env_kwargs, num_envs=1, max_episode_length=max_episode_length, 
                         max_total_reward=max_total_reward, save_folder=save_folder,
                         normalize_observation=False, seed=seed)

        self.lr = lr
        self.discount = discount
        self.epsilon_greedy = epsilon_greedy
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_iterations = num_iterations
        self.verbose = verbose
        self.test_every = test_every
        self.num_test_episodes = num_test_episodes

        agent_kwargs_copy = agent_kwargs.copy()  # To be sure that the original dict is not modified
        agent_kwargs_copy["env_kwargs"] = env_kwargs
        self.current_agent = get_agent(self.obs_space, self.action_space, agent_kwargs_copy, q_table=True)
        self.current_iteration = 0
        self.train_rewards = []
        self.mean_test_rewards = []
        self.std_test_rewards = []
        self.episode_lengths = []

    def train_(self):

        writer = SummaryWriter(os.path.join(self.save_folder, "logs"))

        env = self.make_env().envs[0]

        if self.verbose:
            pbar = trange(self.num_iterations)
        else:
            pbar = range(self.num_iterations)

        for n in pbar:
            
            test_progress = (n+1) % self.test_every == 0
            test_progress += (n+1) == self.num_iterations

            obs, _ = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:

                if np.random.rand() < self.epsilon_greedy:
                    action = env.action_space.sample()
                else:
                    action = self.current_agent.get_action(obs)
                new_obs, reward, done, _, _ = env.step(action)
                episode_reward += reward
                episode_length += 1

                q = self.current_agent.sample(obs, action)
                q_next = np.max(self.current_agent.sample(new_obs))
                new_q = q + self.lr * (reward + self.discount * q_next - q)
                self.current_agent.update(obs, action, new_q)

                obs = new_obs

                if episode_length >= self.max_episode_length and self.max_episode_length != -1:
                    done = True

                if episode_reward >= self.max_total_reward and self.max_total_reward != -1:
                    done = True

            self.train_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            writer.add_scalar("train_reward", episode_reward, self.current_iteration)
            writer.add_scalar("episode_length", episode_length, self.current_iteration)

            self.epsilon_greedy = max(self.epsilon_min, self.epsilon_greedy * self.epsilon_decay)

            self.current_iteration += 1

            if test_progress:
                mean, std = self.test(self.num_test_episodes)
                self.mean_test_rewards.append(mean)
                self.std_test_rewards.append(std)
                writer.add_scalar("test_reward", mean, self.current_iteration)
                writer.add_scalar("test_reward_std", std, self.current_iteration)
                self.save(self.models_folder + f"/iter_{n+1}.pkl")

            if self.verbose:
                description = "Train Reward: {:.2f}, Epsilon: {:.2f}".format(episode_reward, self.epsilon_greedy)
                if len(self.mean_test_rewards) > 0:
                    description += ", Test Reward: {:.2f}".format(self.mean_test_rewards[-1])
                pbar.set_description(description)

    def save(self, path):

        kwargs = self.kwargs.copy()

        running_results = {
            "train_rewards": self.train_rewards,
            "mean_test_rewards": self.mean_test_rewards,
            "std_test_rewards": self.std_test_rewards,
            "episode_lengths": self.episode_lengths,
            "self.current_iteration": self.current_iteration,
        }

        model = {
            "q_table": self.current_agent.q_table
        }

        folders = {
            "save_folder": self.save_folder,
            "models_folder": self.models_folder,
            "plots_folder": self.plots_folder,
            "videos_folder": self.videos_folder
        }

        data = {
            "kwargs": kwargs,
            "running_results": running_results,
            "model": model,
            "folders": folders
        }
        
        np.save(data, path)
    
    def load(self, path, verbose=True):
        
        data = np.load(path, allow_pickle=True).item()

        self.__init__(**data["kwargs"])

        for key in data["running_results"]:
            setattr(self, key, data["running_results"][key])
            
        self.current_agent.q_table = data["model"]["q_table"]

        for key in data["folders"]:
            setattr(self, key, data["folders"][key])

        if verbose:
            print("Loaded model from", path)
            print("Current iteration:", self.current_iteration)
            print("Current epsilon:", self.epsilon_greedy)
            print("Current reward:", self.mean_test_rewards[-1])

    def save_plots(self):

        import matplotlib.pyplot as plt

        plt.figure()
        x_range = np.arange(self.test_every, self.current_iteration+1, self.test_every)
        if len(x_range) == len(self.mean_test_rewards) - 1:
            x_range = np.append(x_range, self.current_iteration)
        plt.plot(x_range, self.mean_test_rewards, label="Mean test reward", c="red")
        plt.fill_between(x_range, np.array(self.mean_test_rewards) - np.array(self.std_test_rewards),
                          np.array(self.mean_test_rewards) + np.array(self.std_test_rewards), alpha=0.2, color="red")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.savefig(self.plots_folder + "/rewards.png")

        plt.figure()
        plt.plot(self.episode_lengths)
        plt.xlabel("Episode")
        plt.ylabel("Episode length")
        plt.savefig(self.plots_folder + "/episode_lengths.png")
