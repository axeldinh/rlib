
import numpy as np
import gymnasium as gym

WARNING = '\033[93m'
ENDC = '\033[0m'

class QTable:
    """
    Q-Table class for classic Q-Learning.

    The Q-Table is a table of size `(grid_size, grid_size, ..., grid_size, action_size)` where `grid_size` is the number of discretization for each dimension of the state space and action_size is the number of actions.
    The values given by the QTable are the Q-values for each state-action pair, i.e.:

    .. math::

        Q(s_t, a_t) = \\sum_{k=0}^{T} \\gamma^{k} R(s_{t+k+1}, a_{t+k+1}) 

    where :math:`\\gamma` is the discount factor, :math:`T` the end of the episode, :math:`s_t` the state at time :math:`t` and :math:`a_t` the action at time :math:`t`.

    Example:

    .. code-block:: python

        from rlib.agents import QTable
        import gymnasium as gym

        env = gym.make("CartPole-v1")

        q_table = QTable(env, grid_size=10)  # 10 discretization for each dimension of the state space
        state = env.reset()
        action = agent.get_action(state)

        q_table.update(state, action, 0.5)  # update the Q-Table with the new value

        q_s_a = q_table.sample(state, action)  # sample the Q-Table for the given state-action pair

        q_s = q_table.sample(state)  # sample the Q-Table for the given state

        best_action = np.argmax(q_s)  # get the best action to take from the given state
        best_action = q_table.get_action(state)  # equivalent to the previous line

    :ivar grid_size: number of discretization for each dimension of the state space.
    :vartype grid_size: int
    :ivar state_size: size of the state space.
    :vartype state_size: int
    :ivar action_size: size of the action space.
    :vartype action_size: int
    :ivar q_table: the Q-Table.
    :vartype q_table: np.ndarray

    """

    def __init__(self, env_kwargs, grid_size=10):
        """
        Initialize the QTable class

        :param env_kwargs: gymnasium environment kwargs, is used to call `gym.make(**env_kwargs)`.
        :param grid_size: number of discretization for each dimension of the state space.
        :type env: dict
        :type grid_size: int, optional
        :raises ValueError: if the action space is not discrete
        
        """

        env = gym.make(**env_kwargs)

        # Check if the action space is discrete
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("The action space must be discrete.")

        # Check if the environment is continuous
        if not isinstance(env.observation_space, gym.spaces.Box):
            self.observation_type = "discrete"
        else:
            self.observation_type = "continuous"

        if self.observation_type == "discrete":
            
            observation_size = []

            if isinstance(env.observation_space, gym.spaces.tuple.Tuple):
                for space in env.observation_space:
                    observation_size.append(space.n)

            else:
                observation_size.append(env.observation_space.n)

            self.q_table = np.random.rand(*tuple(observation_size + [env.action_space.n])) * 2 - 1

        else:

            self.min_values = env.observation_space.low.copy()
            self.max_values = env.observation_space.high.copy()

            # Warning if a value is not bounded
            if np.any(self.min_values <= -1e10) or np.any(self.max_values >= 1e10):
                print(WARNING+"Warning: the environment has infinite state space bounds. The QTable may not work properly.")
                print("Min values: {}".format(self.min_values))
                print("Max values: {}".format(self.max_values))
                print("You may want to use an ObservationWrapper to limit the value range."+ENDC)
            
            self.diff = self.max_values - self.min_values
            self.grid_size = grid_size
            self.state_size = env.observation_space.shape[0]
            self.action_size = env.action_space.n
            self.q_table = np.random.rand(*tuple([self.grid_size] * self.state_size + [self.action_size])) * 2 - 1

    def update(self, state, action, new_value):
        """
        Update the QTable, given a new value for a state-action pair

        :param state: current state
        :param action: action to take
        :param new_value: new value to set
        :type state: np.array
        :type action: int
        :type new_value: float
        
        """
        state_index = self.discretize(state)
        self.q_table[state_index][action] = new_value

    def sample(self, state, action=None):
        """
        Sample the QTable, if no action is given return the Q-values for each action, otherwise return the Q-value for the given action.

        :param state: current state
        :param action: action to take
        :type state: np.array
        :type action: int, optional
        :return: sampled value
        :rtype: float if action is given, np.array otherwise
        
        """

        state_index = self.discretize(state)
        if action is None:
            if self.observation_type == "discrete":
                return self.q_table[tuple(state_index)]
            return self.q_table[state_index]
        else:
            return self.q_table[state_index][action]

    def get_action(self, state):
        """
        Get the action to take from the current state

        :param state: current state
        :type state: np.array
        :return: action to take
        :rtype: int
        
        """
        return np.argmax(self.sample(state))
    
    def discretize(self, state):
        """
        Discretize the state, i.e. convert it to a tuple of integers allowing sampling in the table.

        :param state: current state
        :type state: np.array
        :return: discretized state
        :rtype: tuple
        
        """
        if self.observation_type == "discrete":
            return np.array(state, dtype=int)
        discretized_state = tuple(((state - self.min_values) / self.diff * (self.grid_size-1)).astype(int))
        return discretized_state
