
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