
from gymnasium import ObservationWrapper

class NormWrapper(ObservationWrapper):
    """
    A gymnasium wrapper that normalizes the observations in [-1, 1].
    Unbounded observations are untouched.
    """

    def __init__(self, env):
        super().__init__(env)
        self.min_values = env.observation_space.low
        self.max_values = env.observation_space.high

        normalized_states = (self.min_values >= -1e10) * (self.max_values <= 1e10)

        self.observation_space.low[normalized_states] = -1
        self.observation_space.high[normalized_states] = 1

        self.min_values[self.min_values <= -1e10] = 0
        self.max_values[self.max_values >= 1e10] = 1
        

    def observation(self, observation):
        return (observation - self.min_values) / (self.max_values - self.min_values) * 2 - 1
