
from abc import abstractclassmethod
import os
import numpy as np
from tqdm  import tqdm
from rlib.utils import play_episode

class BaseAlgorithm:
    """
    This class is the base class for all algorithms.

    Once implemented, an algorithm can be used as follow:

    .. code-block:: python

        import gymnasium
        from rlib.learning import Algorithm
        from rlib.agents import Agent

        env_fn = lambda render_mode=None: gymnasium.make('CartPole-v1', render_mode=render_mode)
        agent_fn = Agent
        model = Algorithm(env_fn, agent_fn, **kwargs)  # kwargs are specific to the algorithm

        algorithm.train()
        algorithm.test()

    :ivar env_fn: A function that returns an `gymnasium.ENV` environment. It should take one argument `render_mode`.
    :vartype env_fn: function
    :ivar agent_fn: A function that returns an agent, without arguments. It should have a `get_action(observation)` method.
    :vartype agent_fn: function
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

    """

    def __init__(
            self, env_fn, agent_fn,
            max_episode_length=-1, 
            max_total_reward=-1,
            save_folder="results"
            ):
        """
        Base class for all the algorithms.

        :param env_fn: A function that returns an `gymnasium.ENV` environment. It should take one argument `render_mode`.
        :type env_fn: function
        :param agent_fn: A function that returns an agent, without arguments. It should have a `get_action(observation)` method.
        :type agent_fn: function
        :param max_episode_length: Maximum number of steps taken to complete an episode. Default is -1 (no limit)
        :type max_episode_length: int, optional
        :param max_total_reward: Maximum reward achievable in one episode. Default is -1 (no limit)
        :type max_total_reward: float, optional
        :param save_folder: The path of the folder where to save the results. Default is "results"
        :type save_folder: str, optional

        """

        self.env_fn = env_fn
        self.agent_fn = agent_fn
        self.max_episode_length = max_episode_length
        self.max_total_reward = max_total_reward

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

    def _create_folders(self):
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
        self._create_folders()
        self.train_()

    @abstractclassmethod
    def save(self, path):
        """
        Save the current agent to the given path.

        :param path: The path to save the agent to.
        :type path: str

        """
        raise NotImplementedError
    
    @abstractclassmethod
    def load(self, path):
        """
        Load the agent from the given path. 
        
        Note that only the agent is loaded, not the environment, 
        nor the training parameters.

        :param path: The path to load the agent from.
        :type path: str

        """
        raise NotImplementedError
    
    def save_plots(self):
        """
        Save the plots of the training.

        The plots are saved in the plots folder :attr:`plots_folder`.
        
        """
        raise NotImplementedError
    
    def test(self, num_episodes=1, display=False, save_video=False, video_path=None):
        """
        Test the current agent on the environment.

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
            env = self.env_fn(render_mode="human")
        elif save_video:
            env = self.env_fn(render_mode="rgb_array")
        else:
            env = self.env_fn(render_mode=None)

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
        """
        Save videos of the models saved at testing iterations.

        The videos are saved in the videos of the saving folder :attr:`save_folder`.
        """

        # Random number to avoid overwriting
        key = np.random.randint(0, 1000000)
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