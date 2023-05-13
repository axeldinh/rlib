from rlib.envs.flappy_bird_gymnasium import flappy_bird_env

from gymnasium.envs import register

register(
    id='FlappyBird-v0',
    entry_point='rlib.envs.flappy_bird_gymnasium.flappy_bird_env:FlappyBirdEnv',
)

__version__ = "0.0.1"