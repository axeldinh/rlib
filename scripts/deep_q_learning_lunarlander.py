import numpy as np
from rlib.learning import DeepQLearning

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

NUM_TIMESTEPS = int(500_000)
LEARNING_STARTS = 10_000
SIZE_REPLAY_BUFFER = 100_000
MAIN_TARGET_UPDATE = 100
UPDATE_FREQUENCY = 4
LR = 3e-4
DISCOUNT = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EXPLORATION_FRACTION = 0.1
BATCH_SIZE = 64
TEST_EVERY = max(1, int(NUM_TIMESTEPS / 20))
NUM_TEST_EPISODES = 5
MAX_REWARD = -1
MAX_EPISODE_LENGTH = 1_000
HIDDEN_SIZES = [64, 64]


def train():
    
    model = DeepQLearning(
        env_kwargs={'id': 'LunarLander-v2'},
        agent_kwargs={'hidden_sizes': HIDDEN_SIZES},
        max_episode_length = MAX_EPISODE_LENGTH,
        max_total_reward=-1,
        num_time_steps = NUM_TIMESTEPS,
        learning_starts=LEARNING_STARTS,
        size_replay_buffer=SIZE_REPLAY_BUFFER,
        lr = LR,
        discount = DISCOUNT,
        epsilon_start = EPSILON_START,
        epsilon_min = EPSILON_MIN,
        exploration_fraction = EXPLORATION_FRACTION,
        batch_size = BATCH_SIZE,
        test_every=TEST_EVERY,
        num_test_episodes=NUM_TEST_EPISODES,
        verbose = True,
        main_target_update=MAIN_TARGET_UPDATE,
        update_every=UPDATE_FREQUENCY,
    )
    
    model.train()
    model.test()
    model.save_plots()
    model.save_videos()


if __name__ == "__main__":

    train()
