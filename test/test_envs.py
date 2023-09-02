"""
Here we check that all environments can be imported and activated correctly.
"""

import os
import rlib
from rlib.utils import play_episode
import gymnasium

class bcolors:
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'

try:
    from rlib.envs import *
    print("Imported rlib.envs")
except ImportError as e:
    raise e

def single_test_env(env_name):

    print(f"Testing {env_name} environment")

    try:
        gymnasium.make(env_name, render_mode=None)
    except Exception as e:
        print(f"\tFailed to create {env_name} environment without rendering")
        raise e
    else:
        print(f"\t{bcolors.OKGREEN}Created {env_name} environment without rendering")

    try:
        gymnasium.make(env_name, render_mode="human")
    except Exception as e:
        print(f"\t{bcolors.FAIL}Failed to create {env_name} environment with human rendering")
        raise e
    else:
        print(f"\t{bcolors.OKGREEN}Created {env_name} environment with human rendering")

    try:
        gymnasium.make(env_name, render_mode="rgb_array")
    except Exception as e:
        print(f"\t{bcolors.FAIL}Failed to create {env_name} environment with rgb_array rendering")
        print("\t{bcolors.FAIL}Cannot check video saving without rgb_array rendering")
        raise e
    else:
        print(f"\t{bcolors.OKGREEN}Created {env_name} environment with rgb_array rendering")

    try:
        play_episode(gymnasium.make(env_name, render_mode=None), None, max_episode_length=1)
    except Exception as e:
        print(f"\t{bcolors.FAIL}Failed to play one episode of {env_name} without rendering")
        raise e
    else:
        print(f"\t{bcolors.OKGREEN}Played one episode of {env_name} without rendering")

    try:
        play_episode(gymnasium.make(env_name, render_mode="rgb_array"), None, max_episode_length=1, save_video=True, video_path="test.mp4")
    except Exception as e:
        print(f"\t{bcolors.FAIL}Failed to play one episode of {env_name} and save video")
        raise e
    else:
        print(f"\t{bcolors.OKGREEN}Played one episode of {env_name} and saved video")
        os.remove("test.mp4")

def test_envs():
    
    single_test_env("FlappyBird-v0")