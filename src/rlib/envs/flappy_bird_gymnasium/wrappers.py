import os
import numpy as np
import cv2
from tqdm  import tqdm
from gymnasium import Wrapper
from .renderer import FlappyBirdRenderer


class FutureSaver(Wrapper):

    def __init__(self, env, save=True):
        """
        A gymnasium wrapper for the FlappyBirdEnv that allows the saving
        of the state of the environment at each step. The states are kept in memory
        as a list of dictionaries, where each dictionary contains the following
        keys:
            - bird_pos_y: the y position of the bird
            - velocity_y: the y velocity of the bird
            - last_actions: the last actions taken by the bird
            - pipes: a list of Pipe objects
            - score: the current score of the bird
        Those states can be saved and used in the future to save a video of the
        bird playing the game. This allows the making of a video without having
        the agent to play the game again. This also avoids memory issues when
        saving a video of a long game.

        
        """
        super().__init__(env)
        self.env = env
        self.save = save

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        if self.save:
            self.all_states.append(
                {
                    "bird_pos_y": self.env.bird_pos_y,
                    "velocity_y": self.env.velocity_y,
                    "last_actions": self.env.last_actions,
                    "pipes": [pipe.copy() for pipe in self.env.pipes],
                    "score": self.env.score,
                }
            )
        return state, reward, done, truncated, info
    
    def save_states(self, folder, name):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, name)
        np.save(path, self.all_states)

    def load_states(self, folder, name):
        path = os.path.join(folder, name)
        self.all_states = np.load(path, allow_pickle=True).tolist()

    def save_video(self, folder, name):

        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, name)

        self.renderer = FlappyBirdRenderer(
            self.env, 
            render_mode='rgb_array',
            window_size=self.env.window_size,
            bird_size=self.env.bird_size,
            debug=self.env.debug
            )

        video_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), self.env.metadata["render_fps"], (288, 512))

        print("Saving video...")

        for state in tqdm(self.all_states):
            
            # The environment must update by itself, else the variables will not be updated
            # again in reset or step
            self.env.update_state(state)

            frame = self.renderer.render_frame(mode="rgb_array")
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)

        print("Video saved!")

        video_writer.release()

        self.__init__(self.env, save=self.save)

    def reset(self, *args, **kwargs):

        observation, info = super().reset(*args, **kwargs)
        if self.save:
            self.all_states = [{
                "bird_pos_y": self.env.bird_pos_y,
                "velocity_y": self.env.velocity_y,
                "last_actions": self.env.last_actions,
                "pipes": [pipe.copy() for pipe in self.env.pipes],
                "score": self.env.score,
            }]

        return observation, info