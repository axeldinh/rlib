
import gymnasium
from gymnasium import ObservationWrapper

class NormWrapper(ObservationWrapper):
    """
    A gymnasium wrapper that normalizes the observations in [-1, 1].
    Unbounded observations are untouched.
    """

    def __init__(self, env):
        super().__init__(env)
        self.min_values = env.observation_space.low.copy()
        self.max_values = env.observation_space.high.copy()

        normalized_states = (self.min_values >= -1e10) * (self.max_values <= 1e10)

        low = self.min_values.copy()
        low[normalized_states] = -1
        high = self.max_values.copy()
        high[normalized_states] = 1

        self.observation_space = gymnasium.spaces.Box(
                low=low, 
                high=high,
                shape=self.min_values.shape, dtype=low.dtype)

        self.min_values[~normalized_states] = 0
        self.max_values[~normalized_states] = 1
        

    def observation(self, observation):
        observation = (observation - self.min_values) / (self.max_values - self.min_values) * 2 - 1
        return observation
