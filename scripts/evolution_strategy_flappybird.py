from rlib.learning import EvolutionStrategy

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

NUM_AGENTS = 30
NUM_ITERATIONS = 1000
LR = 0.03
SIGMA = 0.1
TEST_EVERY = 10
MAX_EPISODE_LENGTH = -1
GAP_SIZE = 125
MAX_SCORE = 1000

SAVE_FOLDER = "evolution_strategy"

ENV_NAME = "FlappyBird-v0"
HIDDEN_SIZES = [50]

def train():
    
    model = EvolutionStrategy(
        env_kwargs={'id': ENV_NAME, 'gap': GAP_SIZE, 'observation_mode': 'simple'},
        agent_kwargs={'hidden_sizes': HIDDEN_SIZES},
        num_agents=NUM_AGENTS,
        num_iterations=NUM_ITERATIONS,
        lr=LR,
        sigma=SIGMA,
        test_every=TEST_EVERY,
        max_episode_length=MAX_EPISODE_LENGTH,
        max_total_reward=MAX_SCORE,
        stop_max_score=True,
        save_folder=SAVE_FOLDER,
    )

    model.train()

    model.save_plots()
    model.save_videos()

if __name__ == "__main__":

    train()

