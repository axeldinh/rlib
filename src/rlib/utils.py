
import cv2


def play_episode(env, agent=None, 
                max_episode_length=-1,
                max_total_reward=-1,
                save_video=False,
                video_path=None):
    """
    Plays an episode of the environment using the agent.

    :param env: The environment to play.
    :type env: `gymnasium.ENV`
    :param agent: The agent to use to solve the environment. It should have a `get_action(observation)` method. If None, the actions are sampled randomly.
    :type agent: optional
    :param max_episode_length: The maximum  total reward to get in the episode, by default -1 (no limit).
    :type max_episode_length: int, optional
    :param max_total_reward: The maximum total reward to get in the episode, by default -1 (no limit).
    :type max_total_reward: float, optional
    :param save_video: Whether to save a video of the episode, by default False.
    :type save_video: bool, optional
    :param video_path: The path to save the video, by default None.
    :type video_path: str, optional
    :return: The total reward obtained during the episode.
    :rtype: float

    Example:

    .. code-block:: python

        import gymnasium as gym
        from rlib.utils import play_episode
        env = gym.make("CartPole-v0")
        play_episode(env)

    """

    obs, _ = env.reset()
    total_reward = 0
    episode_length = 0
    done = False

    if save_video:
        frame = env.render()
        frame_shape = frame.shape

        video_writer = cv2.VideoWriter(
            video_path, 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            env.metadata["render_fps"],
            frame_shape[:2][::-1])

    while not done:
        if agent is None:
            action = env.action_space.sample()
        else:
            action = agent.get_action(obs)
        obs, reward, done, _, _  = env.step(action)
        total_reward += reward
        episode_length += 1

        if episode_length >= max_episode_length and max_episode_length != -1:
            done = True

        if total_reward >= max_total_reward and max_total_reward != -1:
            done = True

        if env.render_mode is not None:
            frame = env.render()

        if save_video:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)

    if save_video:
        video_writer.release()
    
    return total_reward, episode_length

def get_git_infos(path=None):
    """ Returns the git infos of the repository and save it
    """
    import subprocess
    import os

    if path is None:
        path = os.path.dirname(os.path.abspath(__file__))

    # Get the remote url
    remote_url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"], 
                                            cwd=path).decode("utf-8").strip()

    # Get the commit hash
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], 
                                            cwd=path).decode("utf-8").strip()

    # Get the commit message
    commit_message = subprocess.check_output(["git", "log", "-1", "--pretty=%B"],
                                                cwd=path).decode("utf-8").strip().replace("|", " ").replace("\n", " ")

    # Get the branch name
    branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                            cwd=path).decode("utf-8").strip()

    with open(os.path.abspath("src/rlib/__git_infos__.py"), "w") as infos_file:
        infos_file.write(f'__remote_url__ = "{remote_url}"\n')
        infos_file.write(f'__commit_hash__ = "{commit_hash}"\n')
        infos_file.write(f'__commit_message__ = "{commit_message}"\n')
        infos_file.write(f'__branch_name__ = "{branch_name}"\n')