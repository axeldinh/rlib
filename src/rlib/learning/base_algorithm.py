
from abc import abstractclassmethod
import os
import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
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
    :ivar num_envs: The number of environments to use for training.
    :vartype num_envs: int
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
    :ivar envs_wrappers: The wrappers to use for the environment, by default None.
    :vartype envs_wrappers: list, optional

    """

    def __init__(
            self, env_kwargs, num_envs,
            max_episode_length=-1, 
            max_total_reward=-1,
            save_folder="results",
            normalize_observation=False,
            seed=42,
            envs_wrappers=None,
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
        :param envs_wrappers: The wrappers to use for the environment, by default None.
        :type envs_wrappers: list, optional

        """

        self.env_kwargs = env_kwargs
        self.num_envs = num_envs
        self.normalize_observation = normalize_observation
        self.max_episode_length = max_episode_length
        self.max_total_reward = max_total_reward
        self.seed = seed
        self.envs_wrappers = envs_wrappers

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        # Determine the actions and observations spaces, useful to know if MLPs or CNNs should be used
        env = self.make_env()
        self.action_space = env.single_action_space
        self.obs_space = env.single_observation_space
        del env

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

        if self.envs_wrappers is not None:
            for wrapper in self.envs_wrappers:
                env = wrapper(env)

        return env

    def _create_folders(self):
        """ Creates the folders for saving the results.
        """
        os.makedirs(self.save_folder, exist_ok=True)
        os.makedirs(self.videos_folder, exist_ok=True)
        os.makedirs(self.models_folder, exist_ok=True)
        os.makedirs(self.plots_folder, exist_ok=True)

    def save_git_info(self):
        """ Saves the git info of the repository.
        """
        import subprocess
        import sys

        abs_path_git_directory = os.path.abspath(__file__)
        # Stop at src in case someone cloned the repo with a different name
        while os.path.basename(os.path.normpath(abs_path_git_directory)) != "src":
            abs_path_git_directory = os.path.dirname(abs_path_git_directory)
        abs_path_git_directory = os.path.dirname(abs_path_git_directory)

        # Get the remote url
        remote_url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"], 
                                             cwd=abs_path_git_directory).decode("utf-8").strip()

        # Get the commit hash
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], 
                                              cwd=abs_path_git_directory).decode("utf-8").strip()

        # Get the commit message
        commit_message = subprocess.check_output(["git", "log", "-1", "--pretty=%B"],
                                                 cwd=abs_path_git_directory).decode("utf-8").strip().replace("|", " ")

        # Get the branch name
        branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                              cwd=abs_path_git_directory).decode("utf-8").strip()

        clone_command = "git clone {}".format(remote_url)
        checkout_command = 'git checkout -b "{}" {}'.format(f"{branch_name}_{commit_hash}", commit_hash)
        command_to_run = "python " + " ".join(sys.argv)

        git_infos = "| RLib Git info | Value |\n"
        git_infos += "| :--- | ---: |\n"
        git_infos += "| Remote url | {} |\n".format(remote_url)
        git_infos += "| Branch name | {} |\n".format(branch_name)
        git_infos += "| Commit hash | {} |\n".format(commit_hash)
        git_infos += "| Commit message | {} |\n".format(commit_message)
        git_infos += "| Clone command | {} |\n".format(clone_command)
        git_infos += "| Checkout command | {} |\n".format(checkout_command)
        git_infos += "| Command to run | {} |\n".format(command_to_run)

        writer = SummaryWriter(os.path.join(self.save_folder, "logs"))
        writer.add_text("RLib Git info", git_infos)
        writer.close()
        

    def save_hyperparameters(self):
        """ Saves the hyperparameters of the algorithm.
        """
        if not hasattr(self, "kwargs"):
            raise ValueError("kwargs should be defined in the __init__ method of the algorithm, this can be done by setting `self.kwargs = locals()` and removing `self``and `__class__`")
        
        writer = SummaryWriter(os.path.join(self.save_folder, "logs"))
        summary = "| Hyperparameter | Value |\n"
        summary += "| :--- | ---: |\n"
        for key, value in self.kwargs.items():
            summary += "| {} | {} |\n".format(key, value)
        writer.add_text("Hyperparameters", summary)
        writer.close()

    @abstractclassmethod
    def train_(self) -> None:
        """
        Train the agent on the environment.

        This method should be implemented in the child class.
        """
        raise NotImplementedError
    
    def train(self):
        """ Default training method.

        Along with the training, it creates the folders for saving the results, saves the hyperparameters and the git info.
        """
        self._create_folders()
        self.save_hyperparameters()
        self.save_git_info()
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
            r, _  = play_episode(
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
            video_path = os.path.join(self.videos_folder, ".".join(model.split('.')[:-1]) + ".mp4")
            if os.path.exists(video_path):
                print(video_path)
                continue
            self.load(os.path.join(self.models_folder, model), verbose=False)
            try:
                self.test(save_video=True, video_path=video_path)
            except Exception as e:
                print("Failed to save video for model {}".format(model))
                print(e)
                continue
        
        print("Videos saved in {}".format(self.videos_folder))

        self.load(f".tmp{key}.pkl", verbose=False)
        os.remove(f".tmp{key}.pkl")
