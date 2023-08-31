import gymnasium
from gymnasium import Env
import numpy as np
import cv2

from .renderer import FlappyBirdRenderer

class Pipe:

    def __init__(self, height_top, height_bottom, pos_x):
        self.height_top = height_top
        self.height_bottom = height_bottom
        self.pos_x = pos_x
        self.passed = False

    def __repr__(self):
        return "Pipe(x: {:.3f}, Gap: {:.3f} - {:.3f})".format(self.pos_x, self.height_top, self.height_bottom)
    
    def copy(self):
        return Pipe(self.height_top, self.height_bottom, self.pos_x)

class FlappyBirdEnv(Env):
    """
    `Gymnasium` environment for the Flappy Bird game.

    The environment is a clone of the original Flappy Bird game, with the same rules and mechanics.

    The observation space is a 8-dimensional vector containing the following values:

    Action Space:
    -------------

    +-----------+------------------------------------+
    | Num       | Action                             |
    +===========+====================================+
    | 0         | No flap                            |
    +-----------+------------------------------------+
    | 1         | Flap                               |
    +-----------+------------------------------------+

    Reward:
    -------
    The reward is 1 if the bird passed a pipe, 0 otherwise.

    Observation:
    ------------

    There are two observation modes: simple and image.

    For simple, the observation is a 8-dimensional vector containing the following values:

    +-----------+------------------------------------+-----------------------------+-----------------------------+
    | Num       | Observation                        | Min                         | Max                         |
    +===========+====================================+=============================+=============================+
    | 0         | Bird Vertical Position             | 0                           | 1                           |
    +-----------+------------------------------------+-----------------------------+-----------------------------+
    | 1         | Bird Vertical Velocity             | -inf                        | inf                         |
    +-----------+------------------------------------+-----------------------------+-----------------------------+
    | 2         | Next Pipe Horizontal Distance      | 0                           | 1                           |
    +-----------+------------------------------------+-----------------------------+-----------------------------+
    | 3         | Next pipe Top Elevation            | 0                           | 1                           |
    +-----------+------------------------------------+-----------------------------+-----------------------------+
    | 4         | Next pipe Bottom Elevation         | 0                           | 1                           |
    +-----------+------------------------------------+-----------------------------+-----------------------------+
    | 5         | Next next pipe Horizontal Distance | 0                           | 1                           |
    +-----------+------------------------------------+-----------------------------+-----------------------------+
    | 6         | Next next pipe Top Elevation       | 0                           | 1                           |
    +-----------+------------------------------------+-----------------------------+-----------------------------+
    | 7         | Next next pipe Bottom Elevation    | 0                           | 1                           |
    +-----------+------------------------------------+-----------------------------+-----------------------------+

    For image, the observation is a 512x288x3 RGB image of the current frame.

    Debug:
    ------

    If debug is set to true, the boxes defining the bird and the pipes will be drawn on the screen,
    as well as the actions taken and the current render fps.

    Example:
    --------

    .. code-block:: python

        import gymnasium
        from rlib.envs import FlappyBird

        env = gymnasium.make("FlappyBird-v0", gap=125, observation_mode="simple", debug=False)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30
          }
    
    window_size = (288, 512)
    distance_pipes = 0.5
    ground = 0.8

    # Position of the bird
    bird_pos_x = 0.3

    # Speeds of the bird and pipes
    horizontal_speed = 120  # In pixels per second
    horizontal_speed /= (metadata["render_fps"] * window_size[0])

    # Downward acceleration of the bird
    gravity = 40  # In pixels per second
    gravity /= (metadata["render_fps"] * window_size[1])

    # Acceleration of the bird when flapping
    action_acceleration = 300 # In pixels per second
    action_acceleration /= (metadata["render_fps"] * window_size[1])

    # Bird and Pipes sizes
    bird_size = (24, 24)
    pipe_width = 52

    # Initial variables, values in reset()
    bird_pos_y = None
    velocity_y = None
    last_actions = None
    pipes = None
    score = None
    _done = False
    _ground_touched = False
    _touches_ceil = False


    def __init__(self, render_mode=None, gap=125, observation_mode="simple", debug=False):
        super().__init__()

        if observation_mode == "image" and render_mode is None:
            raise ValueError("If observation mode is image, render mode must be specified")

        self.render_mode = render_mode
        self.observation_mode = observation_mode
        self.debug = debug

        if self.observation_mode == "simple":
            self.observation_space = gymnasium.spaces.Box(
                low=np.array([0.0, -np.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 
                high=np.array([1.0, np.inf, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                shape=(8,), dtype=np.float32)
        elif self.observation_mode == "image":
            self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=(512, 288, 3), dtype=np.uint8)
            
        self.action_space = gymnasium.spaces.Discrete(2)

        self.gap = gap

        self._init_pipes()

        if self.render_mode is not None:
            self.renderer = FlappyBirdRenderer(self, render_mode=self.render_mode,
                                                window_size=self.window_size,
                                                bird_size=self.bird_size,
                                                debug=self.debug
                                                )
            if self.render_mode == "rgb_array_list":
                self._frames = [self.renderer.render_frame(mode="rgb_array")]

    def _create_random_pipe(self, pos_x):

        pipe_height = np.random.uniform(0.3, 0.6)
        pipe = Pipe(pipe_height - self.gap / self.window_size[1] / 2, pipe_height + self.gap / self.window_size[1] / 2, pos_x)

        return pipe


    def _init_pipes(self):
        
        self.pipes = [self._create_random_pipe(1 + i * self.distance_pipes) for i in range(4)]


    def step(self, action):
        """
        Action: 0 for no flap, 1 for flap
        """

        # Move the bird
        
        if action == 1:
            self.velocity_y = -self.action_acceleration

        self.velocity_y += self.gravity

        self.bird_pos_y += self.velocity_y

        if self.bird_pos_y < 0:
            self._done = True
            self._touches_ceil = True

        if self.bird_pos_y > self.ground:
            self._done = True
            self._ground_touched = True

        # Move the pipes
        for pipe in self.pipes:
            if pipe is None:
                continue
            pipe.pos_x -= self.horizontal_speed

        for i, pipe in enumerate(self.pipes):
            if pipe is None:
                continue
            # If the pipe is to far, remove it and create a new one
            if pipe.pos_x <= -0.2:
                self.pipes[i] = None
                max_pos_x_pipe = max([pipe.pos_x for pipe in self.pipes if pipe is not None])
                next_pipe = self._create_random_pipe(max_pos_x_pipe + self.distance_pipes)
                self.pipes[i] = next_pipe

        # Check if the bird collided with a pipe
        if self._collides():
            self._done = True

        # Store the last actions
        self.last_actions = np.roll(self.last_actions, -1)
        self.last_actions[-1] = action

        if self.render_mode == "rgb_array_list":
            self._frames.append(self.renderer.render_frame(mode="rgb_array"))

        return self._get_state(), self._get_reward(), self._is_done(), self._is_truncated(), self.metadata


    def _get_state(self):
        """
        The observation depends on the observation mode.
        For simple:
            Returns the bird's position and velocity, as well as its distance to the next pipe and
            the next pipe's top and bottom height
        For image:
            Returns the current frame
        """

        if self.observation_mode == "simple":
        
            pipes = sorted([pipe for pipe in self.pipes if pipe is not None], key=lambda pipe: pipe.pos_x)
            next_pipe = None

            for next_id, pipe in enumerate(pipes):
                if pipe.pos_x + self.pipe_width / self.window_size[0] > self.bird_pos_x:
                    next_pipe = pipe
                    break

            next_next_pipe = pipes[next_id + 1]

            if next_pipe is None or next_next_pipe is None:
                raise ValueError("There should be at least 2 pipes after the bird")
            
            if next_pipe.pos_x > 1.:
                next_pipe_dist_x = 1.
                next_pipe_top = 0.
                next_pipe_bottom = 1.
            else:
                next_pipe_dist_x = next_pipe.pos_x - self.bird_pos_x
                next_pipe_top = next_pipe.height_top
                next_pipe_bottom = next_pipe.height_bottom

            if next_next_pipe.pos_x > 1.:
                next_next_pipe_dist_x = 1.
                next_next_pipe_top = 0.
                next_next_pipe_bottom = 1.
            else:
                next_next_pipe_dist_x = next_next_pipe.pos_x - self.bird_pos_x
                next_next_pipe_top = next_next_pipe.height_top
                next_next_pipe_bottom = next_next_pipe.height_bottom

            observations = np.array([
                self.bird_pos_y,
                self.velocity_y,
                next_pipe_dist_x,
                next_pipe_top,
                next_pipe_bottom,
                next_next_pipe_dist_x,
                next_next_pipe_top,
                next_next_pipe_bottom
            ], dtype=np.float32)

            return observations
        
        else:
            frame = self.render(mode="rgb_array")
            return frame
        

    def reset(self, *args, **kwargs):
        """
        Resets the environment
        """
        super().reset(*args, **kwargs)
        self.bird_pos_y = 0.4
        self.velocity_y = 0.0
        self.last_actions = np.zeros(3)
        self._init_pipes()
        self._done = False
        self._ground_touched = False
        self._touches_ceil = False
        self.score = 0

        return self._get_state(), {}


    def _get_reward(self):
        """
        Returns the reward for the current state
        """
        
        if self._is_done():
            return -1

        # If the bird passed a pipe, return 1
        if self._passed_pipe():
            return 1

        # Otherwise, return 0
        return 0
        

    def render(self, mode=None):
        if mode is None:
            mode = self.render_mode
        if mode in ["human", "rgb_array"]:
            return self.renderer.render_frame(mode=mode)
        elif mode == "rgb_array_list":
            return self._frames


    def _collides(self):
        """
        Checks if the bird collided with a pipe
        """

        # Find the nearest pipe
        nearest_pipe = self._get_closest_pipe()

        return self._collides_pipe(nearest_pipe)
    

    def _collides_pipe(self, pipe):

        # Checks that the bird is in the pipe's horizontal range
        if self._bird_in_pipe(pipe):

            # Checks if the bird hit the top or bottom of the pipe
            higher_than_top = self.bird_pos_y < pipe.height_top
            lower_than_bottom = self.bird_pos_y + self.bird_size[0] / self.window_size[1] > pipe.height_bottom

            return higher_than_top or lower_than_bottom
        
        return False
    

    def _get_closest_pipe(self):
        """
        Find the nearest pipe to the bird
        """

        nearest_pipe_distance = np.inf
        for pipe in self.pipes:
            if pipe is None:
                continue
            if np.abs(pipe.pos_x - self.bird_pos_x) < nearest_pipe_distance or np.abs(pipe.pos_x + self.pipe_width / self.window_size[1] - self.bird_pos_x) < nearest_pipe_distance:
                nearest_pipe_distance = min(np.abs(pipe.pos_x - self.bird_pos_x), np.abs(pipe.pos_x + self.pipe_width / self.window_size[1] - self.bird_pos_x))
                nearest_pipe = pipe
        
        return nearest_pipe
    

    def _bird_in_pipe(self, pipe):
        """
        Checks if the bird is inside a pipe's horizontal range
        """

        after_pipe_start = self.bird_pos_x + self.bird_size[0] / self.window_size[0] >= pipe.pos_x
        before_pipe_end = self.bird_pos_x <= pipe.pos_x + self.pipe_width / self.window_size[0]
        
        return after_pipe_start and before_pipe_end


    def _is_done(self):
        return self._done
    

    def _is_truncated(self):
        return False
    

    def _passed_pipe(self):
        if self._is_done():
            return False
        for pipe in self.pipes:
            if self.bird_pos_x > pipe.pos_x + self.pipe_width / self.window_size[0] and not pipe.passed:
                pipe.passed = True
                self.score += 1
                return True


    def close(self):
        if self.render_mode is not None:
            self.renderer.close()

    def update_state(self, state):
            
        self.bird_pos_y = state["bird_pos_y"]
        self.velocity_y = state["velocity_y"]
        self.last_actions = state["last_actions"]
        self.pipes = state["pipes"]
        self.score = state["score"]