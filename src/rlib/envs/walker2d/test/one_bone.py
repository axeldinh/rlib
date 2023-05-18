
import sys
sys.path.append("../")

from rlib.envs.walker2d.bone import Bone
import pygame
import numpy as np

def test_with_force(bone, force, position_force):

    pygame.init()

    screen_size = (500, 500)
    screen = pygame.display.set_mode(screen_size)

    bone.apply_force(cartesian_force=force, position_force=position_force)

    while True:

        screen.fill((0, 0, 0))

        bone.update(0.01)
        bone.draw(screen)

        pygame.display.flip()
        
        # Stop by pressing the close button
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

def horizontal_bone_horizontal_force():

    bone = Bone(
        np.array([0.5, 0.5]),
        np.array([0.6, 0.5]),
        1,
        0.005,
    )
    force = np.array([10, 0])

    test_with_force(bone, force, 0.5)


def horizontal_bone_vertical_force_middle():

    bone = Bone(
        np.array([0.5, 0.5]),
        np.array([0.6, 0.5]),
        1,
        0.005,
    )
    force = np.array([0, 10])

    test_with_force(bone, force, 0.5)

def horizontal_bone_vertical_force_end():

    bone = Bone(
        np.array([0.5, 0.5]),
        np.array([0.6, 0.5]),
        1,
        0.005,
    )
    force = np.array([0, 10])

    test_with_force(bone, force, 1)

def vertical_bone_horizontal_force_start():

    bone = Bone(
        np.array([0.5, 0.5]),
        np.array([0.5, 0.6]),
        1,
        0.005,
    )
    force = np.array([10, 0])

    test_with_force(bone, force, 0.)

def angular_down_both_ends():

    """
    Bone should not move
    """

    bone = Bone(
        np.array([0.5, 0.5]),
        np.array([0.6, 0.5]),
        1,
        0.005,
    )
    bone.apply_force(angular_force=10, position_force=0.)
    bone.apply_force(angular_force=-10, position_force=1.)

    test_with_force(bone, 0, 0.)


if __name__ == "__main__":

    horizontal_bone_horizontal_force()
    horizontal_bone_vertical_force_middle()
    horizontal_bone_vertical_force_end()
    vertical_bone_horizontal_force_start()
    angular_down_both_ends()
