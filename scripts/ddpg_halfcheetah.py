from rlib.learning import DDPG
import numpy as np
from gymnasium.wrappers import ClipAction, TransformObservation, NormalizeReward, TransformReward

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

NUM_EPISODES = 4000
BATCH_SIZE = 128
LEARNING_STARTS = 100_000
TEST_EVERY = 10
MAX_EPISODE_LENGTH = 1000
NUM_TEST_EPISODES = 10
NUM_UPDATES = 10
SIZE_REPLAY_BUFFER = int(1e6)
SEED = 0

if __name__ == "__main__":

    env_kwargs = {'id': 'HalfCheetah-v4'}
    mu_kwargs = {'hidden_sizes': (300, 400)}
    q_kwargs = {'hidden_sizes': (300, 400)}

    ddpg = DDPG(env_kwargs=env_kwargs, mu_kwargs=mu_kwargs, q_kwargs=q_kwargs,
                num_episodes=NUM_EPISODES, learning_starts=LEARNING_STARTS, size_replay_buffer=SIZE_REPLAY_BUFFER,
                test_every=TEST_EVERY, max_episode_length=MAX_EPISODE_LENGTH,
                num_test_episodes=NUM_TEST_EPISODES, use_norm_wrappers=False, normalize_observation=False,
                seed=SEED, batch_size=BATCH_SIZE, num_updates_per_iter=NUM_UPDATES)
    
    envs = ddpg.make_env()
    
    ddpg.train()
    ddpg.save_plots()
    ddpg.save_videos()
