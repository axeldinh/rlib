
from abc import abstractclassmethod
import os
import gymnasium as gym
import numpy as np
import torch
import random
from tqdm  import tqdm
from rlib.utils import play_episode

class BaseAlgorithm:
    """
    This class is the base class for all algorithms.

    Once implemented, an algorithm can be used as follow:

    .. code-block:: python

        import gymnasium
        from rlib.learning import DeepQLearning

        env_kwargs = {"env_name": "CartPole-v1"}
        agent_kwargs = {"hidden_sizes": "[200, 200]"}

        model = DeepQLearning(env_kwargs, agent_kwargs)

        model.train()
        model.test()

    :ivar env_kwargs: The kwargs for calling `gym.make(**env_kwargs, render_mode=render_mode)`.
    :vartype env_kwargs: dict
    :ivar max_episode_length: Maximum number of steps taken to complete an episode.
    :vartype max_episode_length: int
    :ivar max_total_reward: Maximum reward achievable in one episode
    :vartype max_total_reward: float
    :ivar save_folder: The path of the folder where to save the results
    :vartype save_folder: str
    :ivar videos_folder: The path of the folder where to save the videos
    :vartype videos_folder: str
    :ivar models_folder: The path of the folder where to save the models
    :vartype models_folder: str
    :ivar plots_folder: The path of the folder where to save the plots
    :vartype plots_folder: str
    :ivar current_agent: The current agent used by the algorithm
    :ivar env_kwargs: The kwargs for calling `gym.make(**env_kwargs, render_mode=render_mode)`.
    :vartype env_kwargs: dict
    :ivar normalize_observation: Whether to normalize the observation in `[-1, 1]`
    :vartype normalize_observation: bool

    """

    def __init__(
            self, env_kwargs, num_envs,
            max_episode_length=-1, 
            max_total_reward=-1,
            save_folder="results",
            normalize_observation=False,
            seed=42,
            ):
        """
        Base class for all the algorithms.

        :param env_kwargs: The kwargs for calling `gym.make(**env_kwargs, render_mode=render_mode)`.
        :type env_kwargs: dict
        :param num_envs: The number of environments to use for training.
        :type num_envs: int
        :param max_episode_length: Maximum number of steps taken to complete an episode. Default is -1 (no limit)
        :type max_episode_length: int, optional
        :param max_total_reward: Maximum reward achievable in one episode. Default is -1 (no limit)
        :type max_total_reward: float, optional
        :param save_folder: The path of the folder where to save the results. Default is "results"
        :type save_folder: str, optional
        :param normalize_observation: Whether to normalize the observation in `[-1, 1]`. Default is False
        :type normalize_observation: bool, optional
        :param seed: The seed to use for the environment.
        :type seed: int

        """

        self.env_kwargs = env_kwargs
        self.num_envs = num_envs
        self.normalize_observation = normalize_observation
        self.max_episode_length = max_episode_length
        self.max_total_reward = max_total_reward
        self.seed = seed

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        # Determine the actions and observations spaces, useful to know if MLPs or CNNs should be used
        env = self.make_env()
        self.action_space = env.single_action_space
        self.obs_space = env.single_observation_space
        del env

        #if isinstance(self.action_space, gym.spaces.Discrete):
        #    self.action_space_type = "discrete"
        #    self.action_shape = (self.action_space.n,)
        #elif isinstance(self.action_space, gym.spaces.Box):
        #    self.action_space_type = "continuous"
        #    self.action_shape = self.action_space.shape
        #else:
        #    raise ValueError("Unknown action space type, action type is {}".format(type(self.action_space)))

        self.save_folder = save_folder
        run_number = 0
        if os.path.exists(self.save_folder):
            for file in os.listdir(self.save_folder):
                if os.path.isdir(os.path.join(self.save_folder, file)):
                    run_number += 1
        self.save_folder = os.path.join(self.save_folder, "run_{}".format(run_number))
        self.videos_folder = os.path.join(self.save_folder, "videos")
        self.models_folder = os.path.join(self.save_folder, "models")
        self.plots_folder = os.path.join(self.save_folder, "plots")

        self.current_agent = None
    
    def make_env(self, render_mode=None, num_envs=None, seed=None):
        """ Returns an instance of the environment, with the desired render mode.
        :param render_mode: The render mode to use, either `None`, "human" or "rgb_array", by default None.
        :type render_mode: str, optional
        :param num_envs: The number of environments to use for training, by default None (uses `self.num_envs`).
        :type num_envs: int, optional
        """

        if num_envs is None:
            num_envs = self.num_envs

        env = gym.vector.SyncVectorEnv(
            [lambda: gym.make(**self.env_kwargs, render_mode=render_mode)] * num_envs
        )

        if seed is not None:
            env.reset(seed=seed)
        else:
            env.reset(seed=self.seed)

        # Keep track of statistics
        env = gym.wrappers.RecordEpisodeStatistics(env)

        if self.normalize_observation:
            env = gym.wrappers.NormalizeObservation(env)

        return env

    def _create_folders(self):
        """ Creates the folders for saving the results.
        """
        os.makedirs(self.save_folder, exist_ok=True)
        os.makedirs(self.videos_folder, exist_ok=True)
        os.makedirs(self.models_folder, exist_ok=True)
        os.makedirs(self.plots_folder, exist_ok=True)

    @abstractclassmethod
    def train_(self) -> None:
        """
        Train the agent on the environment.
        """
        raise NotImplementedError
    
    def train(self):
        """ Default training method, with a sanity on the creation of the folders.
        """
        self._create_folders()
        self.train_()

    @abstractclassmethod
    def save(self, path):
        """ Save the current agent to the given path.

        :param path: The path to save the agent to.
        :type path: str

        """
        raise NotImplementedError
    
    @abstractclassmethod
    def load(self, path):
        """ Load the agent from the given path. 
        
        Note that only the agent is loaded, not the environment, 
        nor the training parameters.

        :param path: The path to load the agent from.
        :type path: str

        """
        raise NotImplementedError
    
    def save_plots(self):
        """ Save the plots of the training.

        The plots are saved in the plots folder :attr:`plots_folder`.
        
        """
        raise NotImplementedError
    
    def test(self, num_episodes=1, display=False, save_video=False, video_path=None):
        """ Test the current agent on the environment.

        :param num_episodes: The number of episodes to test the agent on, by default 1.
        :type num_episodes: int, optional
        :param display: Whether to display the game, by default False.
        :type display: bool, optional
        :param save_video: Whether to save a video of the game, by default False.
        :type save_video: bool, optional
        :param video_path: The path to save the video to, by default None.
        :type video_path: str, optional
        :return: The mean reward obtained over the episodes, and the standard deviation of the reward obtained over the episodes.
        :rtype: float, float
        :raises ValueError: If num_episodes is not strictly positive.

        """

        if num_episodes <= 0:
            raise ValueError("num_episodes should be strictly positive")
        
        if display and save_video:
            raise ValueError("display and save_video cannot be True at the same time")
        
        if display:
            env = self.make_env(render_mode="human", num_envs=1)
        elif save_video:
            env = self.make_env(render_mode="rgb_array", num_envs=1)
        else:
            env = self.make_env(render_mode=None, num_envs=1)

        env = env.envs[0]

        seed = np.random.randint(0, np.iinfo(np.int32).max)
        env.reset(seed=seed)  # Random seed to avoid overfitting to the same initial state
        rewards = []

        for _ in range(num_episodes):
            r  = play_episode(
                env=env, agent=self.current_agent, 
                max_episode_length=self.max_episode_length, 
                max_total_reward=self.max_total_reward,
                save_video=save_video, video_path=video_path)
            rewards.append(r)

        env.close()

        return np.mean(rewards), np.std(rewards)

    def save_videos(self):
        """ Saves videos of the models saved at testing iterations.

        The videos are saved in the saving folder :attr:`save_folder`.
        """

        # Random number to avoid overwriting when multiple instances are running
        key = np.random.randint(0, 1000000)
        # Saving the current agent
        self.save(f".tmp{key}.pkl")

        print("Saving videos from previous iterations...")

        pbar = tqdm(os.listdir(self.models_folder))
        for model in pbar:
            video_path = os.path.join(self.videos_folder, model[:-4] + ".mp4")
            if os.path.exists(video_path):
                continue
            self.load(os.path.join(self.models_folder, model), verbose=False)
            try:
                self.test(save_video=True, video_path=os.path.join(self.videos_folder, model[:-4] + ".mp4"))
            except Exception as e:
                print("Failed to save video for model {}".format(model))
                print(e)
                continue
        
        print("Videos saved in {}".format(self.videos_folder))

        self.load(f".tmp{key}.pkl", verbose=False)
        os.remove(f".tmp{key}.pkl")

    @abstractclassmethod
    def load_model_parameters(self, data):
        """ Load the model parameters from the given data.

        :param data: The data to load the model parameters from should be the data contained in the file saved when :func:`save` is used.
        :type data: dict

        """
        raise NotImplementedError
    