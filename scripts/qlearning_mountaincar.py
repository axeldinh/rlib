import os
import numpy as np
from rlib.learning import QLearning

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

NUM_ITERATIONS = 100_000
MAX_EPISODE_LENGTH = 200
GRID_SIZE = 100
TEST_EVERY = 1000
LR = 0.1
DISCOUNT = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = np.exp(np.log(0.01 / EPSILON) / NUM_ITERATIONS)
NUM_TEST_EPISODES = 10
SAVE_FOLDER = "qlearning"

ENV_NAME = 'MountainCar-v0'

env_kwargs = {'id': ENV_NAME}
agent_kwargs = {'grid_size': GRID_SIZE}

def train():

    model = QLearning(
        env_kwargs, agent_kwargs,
        max_episode_length=MAX_EPISODE_LENGTH,
        num_iterations=NUM_ITERATIONS,
        test_every=TEST_EVERY,
        lr=LR,
        discount=DISCOUNT,
        epsilon_greedy=EPSILON,
        epsilon_decay=EPSILON_DECAY,
        epsilon_min=EPSILON_MIN,
        save_folder=SAVE_FOLDER,
        num_test_episodes=NUM_TEST_EPISODES,
    )

    model.train()

    mean, std = model.test()

    print(f"Final mean reward: {mean:.2f} +- {std:.2f}")

    model.save_plots()
    model.save_videos()

    return model


if __name__ == "__main__":

    model = train()

    # Because the environment is has only two dimensions, we can plot the Q-table as a heatmap
    
    q_table = model.current_agent.q_table
    q_table = q_table.reshape((GRID_SIZE, GRID_SIZE, 3))
    q_table = np.argmax(q_table, axis=2).T

    import matplotlib.pyplot as plt

    plt.imshow(q_table)
    # Set the colorbar ticks as "Left", "Do Nothing", "Right"
    cbar = plt.colorbar(ticks=[0, 1, 2], orientation="horizontal")
    cbar.ax.set_xticklabels(["Left", "Do Nothing", "Right"])
    # Set the colorbar limits
    plt.clim(0, 2)

    # x axis is position in [-1.2, 0.6], y axis is velocity in [-0.7, 0.7]
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    # Set the ticks
    plt.xticks(np.linspace(0, GRID_SIZE, 5), [f"{x:.2f}" for x in np.linspace(-1.2, 0.6, 5)])
    plt.yticks(np.linspace(0, GRID_SIZE, 5), [f"{x:.2f}" for x in np.linspace(-0.7, 0.7, 5)])
    plt.savefig(os.path.join(model.plots_folder, "q_table_actions.png"))
    plt.close()

    # Because the environment is has only two dimensions, we can plot the Q-table as a heatmap
    
    q_table = model.current_agent.q_table
    q_table = q_table.reshape((GRID_SIZE, GRID_SIZE, 3))
    q_table = np.exp(q_table) / np.sum(np.exp(q_table), axis=2, keepdims=True)
    q_table = q_table[:, :, 0].T

    import matplotlib.pyplot as plt

    plt.imshow(q_table)
    cbar = plt.colorbar()
    # Set the colorbar limits
    plt.clim(0, 1)

    # x axis is position in [-1.2, 0.6], y axis is velocity in [-0.7, 0.7]
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    # Set the ticks
    plt.xticks(np.linspace(0, GRID_SIZE, 5), [f"{x:.2f}" for x in np.linspace(-1.2, 0.6, 5)])
    plt.yticks(np.linspace(0, GRID_SIZE, 5), [f"{x:.2f}" for x in np.linspace(-0.7, 0.7, 5)])
    plt.savefig(os.path.join(model.plots_folder, "q_table_left.png"))
    plt.close()

    # Because the environment is has only two dimensions, we can plot the Q-table as a heatmap

    q_table = model.current_agent.q_table
    q_table = q_table.reshape((GRID_SIZE, GRID_SIZE, 3))
    q_table = np.exp(q_table) / np.sum(np.exp(q_table), axis=2, keepdims=True)
    q_table = q_table[:, :, 1].T

    plt.imshow(q_table)
    cbar = plt.colorbar()
    # Set the colorbar limits
    plt.clim(0, 1)

    # x axis is position in [-1.2, 0.6], y axis is velocity in [-0.7, 0.7]
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    # Set the ticks
    plt.xticks(np.linspace(0, GRID_SIZE, 5), [f"{x:.2f}" for x in np.linspace(-1.2, 0.6, 5)])
    plt.yticks(np.linspace(0, GRID_SIZE, 5), [f"{x:.2f}" for x in np.linspace(-0.7, 0.7, 5)])
    plt.savefig(os.path.join(model.plots_folder, "q_table_nothing.png"))
    plt.close()

    # Because the environment is has only two dimensions, we can plot the Q-table as a heatmap

    q_table = model.current_agent.q_table
    q_table = q_table.reshape((GRID_SIZE, GRID_SIZE, 3))
    q_table = np.exp(q_table) / np.sum(np.exp(q_table), axis=2, keepdims=True)
    q_table = q_table[:, :, 2].T

    plt.imshow(q_table)
    cbar = plt.colorbar()
    # Set the colorbar limits
    plt.clim(0, 1)

    # x axis is position in [-1.2, 0.6], y axis is velocity in [-0.7, 0.7]
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    # Set the ticks
    plt.xticks(np.linspace(0, GRID_SIZE, 5), [f"{x:.2f}" for x in np.linspace(-1.2, 0.6, 5)])
    plt.yticks(np.linspace(0, GRID_SIZE, 5), [f"{x:.2f}" for x in np.linspace(-0.7, 0.7, 5)])
    plt.savefig(os.path.join(model.plots_folder, "q_table_right.png"))
    plt.close()
