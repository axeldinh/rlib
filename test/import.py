"""
Here we test that all class and method can be imported correctly.
"""

try:
    import rlib
    print("Imported rlib")
    from rlib.learning import *
    print("Imported rlib.learning")
    from rlib.envs import *
    print("Imported rlib.envs")
    from rlib.utils import *
    print("Imported rlib.utils")
    from rlib.agents import *
    print("Imported rlib.agents")
    from rlib.wrappers import *
    print("Imported rlib.wrappers")
except ImportError as e:
    raise e

import gymnasium
env = gymnasium.make("FlappyBird-v0", render_mode=None)

play_episode(env, None)