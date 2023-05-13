import pygame
import numpy as np
import os
from pathlib import Path

_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
IMG_PATH = os.path.join(_DIR, "assets")

class FlappyBirdRenderer:

    def __init__(self, game, render_mode="human", window_size=(288, 512), bird_size=(20, 20), debug=False):
        
        pygame.init()

        self.game = game
        self.render_mode = render_mode
        self.window_size = window_size
        self.bird_size = bird_size
        self.debug = debug

        pygame.display.init()
        self._load_images()
        if self.render_mode == "human":
            self.window = pygame.display.set_mode(self.window_size)
        
        if debug:
            self.clock = pygame.time.Clock()

    def _load_images(self):
        self.bird_images = [pygame.image.load(os.path.join(IMG_PATH, "yellowbird-downflap.png")),
                            pygame.image.load(os.path.join(IMG_PATH, "yellowbird-midflap.png")),
                            pygame.image.load(os.path.join(IMG_PATH, "yellowbird-upflap.png"))]
        self.bird_images = [pygame.transform.scale(image, self.bird_size) for image in self.bird_images]
        self.bottom_pipe_image = pygame.image.load(os.path.join(IMG_PATH, "pipe-green.png"))
        self.top_pipe_image = pygame.transform.rotate(self.bottom_pipe_image, 180)
        self.background_image = pygame.image.load(os.path.join(IMG_PATH, "background-day.png"))
        self.base_image = pygame.image.load(os.path.join(IMG_PATH, "base.png"))
        self.numbers = [pygame.image.load(os.path.join(IMG_PATH, "{}.png".format(i))) for i in range(10)]

    def render_frame(self, mode="human"):

        canvas = pygame.Surface(self.window_size)
        
        # Draw the background
        canvas.blit(self.background_image, (0, 0))

        # Draw the bird
        last_action = self.game.last_actions[-1]
        last_last_action = self.game.last_actions[-2]
        last_last_last_action = self.game.last_actions[-3]
        if last_action == 1:
            bird_image = self.bird_images[2]
        else:
            if last_last_action == 1:
                bird_image = self.bird_images[1]
            else:
                if last_last_last_action == 1:
                    bird_image = self.bird_images[0]
                else:
                    bird_image = self.bird_images[1]
        bird_image = pygame.transform.rotate(bird_image, -self.game.velocity_y * 200)
        canvas.blit(bird_image, (int(self.game.bird_pos_x * 288), int(self.game.bird_pos_y * 512)))

        # Draw the pipes
        for pipe in self.game.pipes:
            if pipe is None:
                continue
            if pipe.pos_x < -0.3:
                continue
            if pipe.pos_x > 1.3:
                continue
            canvas.blit(self.top_pipe_image, (int(pipe.pos_x * 288), int(pipe.height_top * self.window_size[1]) - self.top_pipe_image.get_height()))
            canvas.blit(self.bottom_pipe_image, (int(pipe.pos_x * 288), int(pipe.height_bottom * self.window_size[1])))

        # Draw the base
        canvas.blit(self.base_image, (0, self.window_size[1] * self.game.ground))

        if self.debug:

            # Draw the action on the top left corner
            if self.game.last_actions[-1] == 1:
                action = "Jump"
            else:
                action = "Fall"
            action_text = pygame.font.SysFont("Arial", 20).render(action, True, (255, 0, 0))
            canvas.blit(action_text, (0, 0))

            pipe_width = self.top_pipe_image.get_width()

            # Draw a red rectangle around the bird
            bird_rect = pygame.Rect(int(self.game.bird_pos_x * 288), int(self.game.bird_pos_y * 512), *self.bird_size)
            pygame.draw.rect(canvas, (255, 0, 0), bird_rect, 1)

            # Draw a blue rectangle for the next pipe and a red one for the next next pipe
            pipes = sorted([pipe for pipe in self.game.pipes if pipe is not None], key=lambda pipe: pipe.pos_x)
            next_pipe = None

            for next_id, pipe in enumerate(pipes):
                if pipe.pos_x + pipe_width / self.window_size[0] > self.game.bird_pos_x:
                    next_pipe = pipe
                    break
            
            next_next_pipe = pipes[next_id + 1]

            next_pipe_rect = pygame.Rect(int(next_pipe.pos_x * 288), int(next_pipe.height_top * self.window_size[1]) - self.top_pipe_image.get_height(), pipe_width, self.top_pipe_image.get_height())
            next_next_pipe_rect = pygame.Rect(int(next_next_pipe.pos_x * 288), int(next_next_pipe.height_top * self.window_size[1]) - self.top_pipe_image.get_height(), pipe_width, self.top_pipe_image.get_height())
            pygame.draw.rect(canvas, (0, 0, 255), next_pipe_rect, 2)
            pygame.draw.rect(canvas, (255, 0, 0), next_next_pipe_rect, 2)

            # Draw the FPS
            fps_text = pygame.font.SysFont("Arial", 20).render("FPS: {:.1f}".format(self.clock.get_fps()), True, (255, 0, 0))
            canvas.blit(fps_text, (0, 0.96 * self.window_size[1]))

            self.clock.tick()

        # Draw the score
        num_digits = len(str(self.game.score))
        first_digit_x = (self.window_size[0] - num_digits * self.numbers[0].get_width()) / 2
        for i, digit in enumerate(str(self.game.score)):
            canvas.blit(self.numbers[int(digit)], (first_digit_x + i * self.numbers[0].get_width(), 0.04 * self.window_size[1]))

        if mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
        else:
            array = pygame.surfarray.array3d(canvas)
            return np.transpose(array, (1, 0, 2))


    def close(self):
        pygame.display.quit()
        pygame.quit()