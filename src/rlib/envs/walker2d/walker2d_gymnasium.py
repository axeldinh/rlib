from gymnasium import Env
import pygame
from rlib.envs.walker2d.bone import Bone
from rlib.envs.walker2d.joint import Joint
from rlib.envs.walker2d.muscle import Muscle
from rlib.envs.walker2d.simulator import Simulator

class Walker2D(Env):

    def __init__(self):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def get_state(self):
        pass

    def get_reward(self):
        pass

    def get_done(self):
        pass

    def get_info(self):
        pass
        



if __name__ == "__main__":

    pygame.init()

    screen_size = (500, 500)

    screen = pygame.display.set_mode(screen_size)

    
    import numpy as np
    
    length = 0.1
    start = (0.5, 0.5)
    bones = []
    for i in range(3):
        if i == 0:
            color = (0, 255, 0)
        else:
            color = (255, 255, 255)
        angle = np.random.rand() * np.pi
        end = (length*np.cos(angle) + start[0], length*np.sin(angle) + start[1])
        bones.append(Bone(start, end, 1, 0.005, color=color))
        start = end

    muscles = []
    joints = []
    for i in range(len(bones)-1):
        muscles.append(Muscle(bones[i], bones[i+1], 0.005, max_angle=340, orientation=-1))
        joints.append(Joint(bones[i], bones[i+1]))

    sim = Simulator(bones, joints, muscles, None, debug=True)

    while True:


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        try:
            if sim.current_time == 0.01:
                print("Second step")
            sim.step(np.random.rand(len(muscles)) * 100)
        except AssertionError:
            print("Assertion error at time: {}".format(sim.current_time))
            input()
            pass
        sim.draw(screen)

        pygame.display.update()


        # Pause until a key is pressed
        #input()

        #  delete the display
        screen.fill((0, 0, 0))
