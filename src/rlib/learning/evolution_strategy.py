
import os
import pickle
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
import torch

from .base_algorithm import BaseAlgorithm
from rlib.utils import play_episode
from rlib.wrappers import NormWrapper

class EvolutionStrategy(BaseAlgorithm):
    """ Implementation of the Evolution Strategy algorithm.

    This algorithm does not need gradient computation, it is therefore
    compatible with any agent.

    The update rule of the weights is given by:

    .. math::
        \\theta_{t+1} = \\theta_t + \\frac{1}{N \sigma} \sum_{i=1}^N r_i \epsilon_i

    where :math:`w_t` are the weights at iteration :math:`t`, :math:`N` is the number of agents,
    :math:`\sigma` is the noise standard deviation, :math:`r_i` is the reward obtained by the agent
    with weights :math:`w_t` + :math:`\sigma\epsilon_i`.

    Each :math:`\epsilon_i` is sampled from a normal distribution with mean 0 and standard deviation 1.

    Examples

    .. code-block:: python

        import gymnasium as gym
        from rlib.learning import EvolutionStrategy
        import numpy as np

        class Agent:

            def __init__(self, params=None):
                if params is None:
                    self.params = np.random.rand(4) * 2 - 1
                else:
                    self.params = params.copy()

            def get_action(self, observation):
                return 1 if np.dot(self.params, observation) > 0 else 0
            
            def set_params(self, params):
                self.params = params["params"].copy()

            def get_params(self):
                return {"params": self.params.copy()}
                    

        env_fn = lambda render_mode: gym.make("CartPole-v0", render_mode=render_mode)
        agent_fn = Agent
        model = EvolutionStrategy(env_fn, agent_fn)
        model.train()
        model.test()
        model.save_plots()
        model.save_videos()

    """
    
    def __init__(
            self, env_fn, agent_fn, num_agents=30,
            num_iterations=300, lr=0.03, sigma=0.1,
            test_every=50, num_test_episodes=5, max_episode_length=-1, 
            max_total_reward=-1, save_folder="evolution_strategy",
            stop_max_score=False,
            verbose=True
            ):
        """
        Initialize the Evolution Strategy algorithm.

        :param env_fn: A function that returns an `gymnasium.ENV` environment. It should take one argument `render_mode`.
        :type env_fn: function
        :param agent_fn: A function that returns an agent, without arguments. It should have a `get_action(observation)` method. Here, the agent should also have a `set_params(params)` method and a `get_params()` method, where `params` is a dictionary of parameters (like in `PyTorch`).
        :type agent_fn: function
        :param num_agents: The number of agents to use to compute the gradient, by default 30
        :type num_agents: int, optional
        :param num_iterations: The number of iterations to run the algorithm, by default 300
        :type num_iterations: int, optional
        :param lr: The learning rate, by default 0.03
        :type lr: float, optional
        :param sigma: The noise standard deviation, by default 0.1
        :type sigma: float, optional
        :param test_every: The number of iterations between each test, by default 50
        :type test_every: int, optional
        :param num_test_episodes: The number of episodes to play during each test, by default 5
        :type num_test_episodes: int, optional
        :param max_episode_length: The maximum number of steps to play in an episode, by default -1 (no limit)
        :type max_episode_length: int, optional
        :param max_total_reward: The maximum total reward to get in an episode, by default -1 (no limit)
        :type max_total_reward: int, optional
        :param save_folder: The folder where to save the models at each test step, by default "evolution_strategy"
        :type save_folder: str, optional
        :param stop_max_score: Whether to stop the training when the maximum score is reached on a test run, by default False
        :param verbose: Whether to display a progression bar during training, by default True
        :type verbose: bool, optional
        :raises ValueError: If the parameters of the agent are not `torch.Tensor` or `np.ndarray`
        :raises ValueError: If the `agent.get_params()` do not return a dictionary.
        :raises ValueError: If the agent does not have a `set_params` method.
        :raises ValueError: If the agent does not have a `get_params` method.

        """

        # Normalize the observation in [-1, 1]
        norm_env_fn = lambda render_mode: NormWrapper(env_fn(render_mode))

        super().__init__(norm_env_fn, agent_fn, max_episode_length=max_episode_length, 
                         max_total_reward=max_total_reward, save_folder=save_folder)

        self.num_agents = num_agents
        self.num_iterations = num_iterations
        self.lr = lr
        self.sigma = sigma
        self.test_every = test_every
        self.num_test_episodes = num_test_episodes
        self.stop_max_score = stop_max_score
        self.verbose = verbose

        self.current_iteration = 0
        self.current_agent = agent_fn()
        self.mean_train_rewards = []
        self.std_train_rewards = []
        self.mean_test_rewards = []
        self.std_test_rewards = []
        
        # Check that agent has a set_params method
        if not hasattr(self.current_agent, "set_params"):
            raise ValueError("The agent should have a set_params method")
        
        # Check that agent has a get_params method
        if not hasattr(self.current_agent, "get_params"):
            raise ValueError("The agent should have a get_params method")

        dummy = self.current_agent.get_params()

        if not isinstance(dummy, dict):
            raise ValueError("The parameters should be a dictionary")

        dummy = dummy[list(dummy.keys())[0]]
        tensor_type = type(dummy)
        
        if tensor_type == torch.Tensor:
            self.tensor_type = "torch"
        elif tensor_type == np.ndarray:
            self.tensor_type = "numpy"
        else:
            raise ValueError("The parameters should be either torch.Tensor or np.ndarray")
        
    def _get_random_parameters(self):
        """
        Returns some randomly generated parameters sampled from normal distribution

        :return: The randomly generated parameters.
        :rtype: dict[str, torch.Tensor]
        """

        agent_params = self.current_agent.get_params()
        noise = {}

        for k in agent_params:
            if self.tensor_type == "torch":
                noise[k] = torch.randn_like(agent_params[k])
            elif self.tensor_type == "numpy":
                noise[k] = np.random.randn(*agent_params[k].shape)

        return noise

    def _parameters_update(self, params, test_rewards, test_noise, lr, sigma):
        """
        Computes the new parameters of the agent.

        The new parameters are given by the formula in :class:`EvolutionStrategy`.

        :param params: The current parameters of the agent.
        :type params: dict[str, torch.Tensor]
        :param test_rewards: The rewards obtained by the agents with the current parameters and the noise.
        :type test_rewards: list[float]
        :param test_noise: The noise used to compute the gradient.
        :type test_noise: dict[str, torch.Tensor]
        :param lr: The learning rate.
        :type lr: float
        :param sigma: The noise standard deviation.
        :type sigma: float
        :return: The new parameters of the agent.
        :rtype: dict[str, torch.Tensor]

        """

        num_agents = len(test_rewards)
        new_params = {}

        for k in params:
            
            if self.tensor_type == "numpy":
                # function to make a torch.Tensor
                # avoids torch copy warning
                f = lambda x: torch.tensor(x)
            else:
                f = lambda x: x
            stack_params = torch.stack([f(noise[k]) for noise in test_noise])
            stack_params_t = stack_params.permute(tuple(i for i in range(1,stack_params.ndim))+(0,))
            dot_product = stack_params_t @ torch.tensor(test_rewards, dtype=stack_params.dtype)
            if self.tensor_type == "torch":
                new_params[k] = params[k] + lr / (num_agents * sigma) * dot_product
            elif self.tensor_type == "numpy":
                new_params[k] = params[k] + lr / (num_agents * sigma) * dot_product.numpy()

        return new_params
    
    def _get_test_parameters(self, params, sigma, noise):
        """
        Given the current parameters, the noise standard deviation and the noise, returns the parameters to test.

        :param params: The current parameters of the agent.
        :type params: dict[str, torch.Tensor]
        :param sigma: The noise standard deviation.
        :type sigma: float
        :param noise: The noise to add to the parameters.
        :type noise: dict[str, torch.Tensor]
        :return: The parameters to test.
        :rtype: dict[str, torch.Tensor]

        """

        test_params = {}
        for k in params:
            test_params[k] = params[k] + sigma * noise[k]

        return test_params

    def train_(self):

        env = self.env_fn(render_mode=None)
        
        params_agent = self.current_agent.get_params()
        agent = self.agent_fn()
        agent.set_params(params_agent)

        if self.verbose:
            progression_bar = trange(self.current_iteration, self.current_iteration + self.num_iterations)
        else:
            progression_bar = range(self.current_iteration, self.current_iteration + self.num_iterations)

        for n in progression_bar:

            test_progress = (n+1) % self.test_every == 0
            test_progress += (n+1) == self.num_iterations

            train_rewards = np.zeros(self.num_agents)
            train_noise = [self._get_random_parameters() for _ in range(self.num_agents)]

            if self.verbose:
                progression_bar2 = trange(self.num_agents, leave=False)
            else:
                progression_bar2 = range(self.num_agents)

            for i in progression_bar2:

                agent.set_params(self._get_test_parameters(params_agent, self.sigma, train_noise[i]))
                train_rewards[i] = play_episode(
                    env=env, agent=agent, 
                    max_episode_length=self.max_episode_length, 
                    max_total_reward=self.max_total_reward)

            mean_reward = np.mean(train_rewards)
            self.mean_train_rewards.append(mean_reward)

            std_reward = np.std(train_rewards)
            self.std_train_rewards.append(std_reward)

            if std_reward > 1e-6:

                rewards_normalized = (train_rewards - mean_reward) / std_reward
                new_params = self._parameters_update(params_agent, rewards_normalized, train_noise, self.lr, self.sigma)
                agent.set_params(new_params)
                params_agent = new_params
                self.current_agent.set_params(params_agent)

            self.current_iteration += 1

            if test_progress:
                mean_test_r, std_test_r = self.test(num_episodes=self.num_test_episodes)
                self.mean_test_rewards.append(mean_test_r)
                self.std_test_rewards.append(std_test_r)
                model_saving_path = os.path.join(self.models_folder, f"iteration_{self.current_iteration}.pkl")
                self.save(model_saving_path)

            if self.verbose:
                description = f"Best Train Reward: {np.max(train_rewards):.2f}, Mean Train Reward: {mean_reward:.2f} (+/- {std_reward:.2f})"
                if self.mean_test_rewards.__len__() > 0:
                    description += f", Test Reward: {self.mean_test_rewards[-1]:.2f}"
                progression_bar.set_description(description)

            if len(self.mean_test_rewards) > 0 and self.stop_max_score:
                if self.mean_test_rewards[-1] == self.max_total_reward:
                    break

        env.close()

    def save(self, path):

        to_save = {
            "num_agents": self.num_agents,
            "num_iterations": self.num_iterations,
            "lr": self.lr,
            "sigma": self.sigma,
            "test_every": self.test_every,
            "max_episode_length": self.max_episode_length,
            "max_total_reward": self.max_total_reward,
            "save_folder": self.save_folder,
            "verbose": self.verbose,
            "current_iteration": self.current_iteration,
            "mean_train_rewards": self.mean_train_rewards,
            "std_train_rewards": self.std_train_rewards,
            "mean_test_rewards": self.mean_test_rewards,
            "std_test_rewards": self.std_test_rewards,
            "current_agent": self.current_agent
        }

        with open(path, "wb") as f:
            pickle.dump(to_save, f)
    
    def load(self, path, verbose=True):

        with open(path, "rb") as f:
            loaded = pickle.load(f)
        
        for key in loaded:
            setattr(self, key, loaded[key])

        if verbose:
            print("Loaded EvolutionStrategy model from {}".format(path))
            print("Current Iteration: {}".format(self.current_iteration))
            print("Best Test Reward: {}".format(max(self.mean_test_rewards)))
            print("Last Test Reward: {}".format(self.mean_test_rewards[-1]))

    def save_plots(self):

        path = os.path.join(self.save_folder, "plots")
        os.makedirs(path, exist_ok=True)

        x = np.arange(1, self.current_iteration+1)

        plt.plot(x, self.mean_train_rewards, label="Mean Train Reward", c="blue")
        plt.fill_between(x, np.array(self.mean_train_rewards) - np.array(self.std_train_rewards), np.array(self.mean_train_rewards) + np.array(self.std_train_rewards), alpha=0.2, color="blue")
        test_range = x[self.test_every-1::self.test_every]
        if len(test_range) == len(self.mean_test_rewards) - 1:
            test_range = np.append(test_range, x[-1])
        plt.plot(test_range, self.mean_test_rewards, label="Test Reward", c="red")
        plt.fill_between(test_range, np.array(self.mean_test_rewards) - np.array(self.std_test_rewards), np.array(self.mean_test_rewards) + np.array(self.std_test_rewards), alpha=0.2, color="red")
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        plt.legend()
        plt.savefig(os.path.join(path, "rewards.png"))
        plt.close()

        print("Saved plots in {}".format(path))
