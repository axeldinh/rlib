from rlib.agents.q_table import QTable
from rlib.agents.mlp import MLP

from gymnasium.spaces import Discrete, MultiDiscrete, Box

def get_agent(obs_space, action_space, kwargs, q_table=False, ddpg_q_agent=False, ppo_critic=False):
    """ Global function to get an agent from its type and parameters

    This is the function used in the algorithms when a kwargs for an agent are given.
    The agent type (MLP, CNN...) is usually automatically inferred from the environment's observation and action spaces.

    Example:

    >>> from rlib.agents import get_agent
    >>> import gymnasium as gym
    >>>
    >>> env = gym.make("CartPole-v1") 
    >>> # This environment has a Box(4,) observation space and a Discrete(2,) action space
    >>> # Hence the infered agent type is a MLP with `input_size=4` and `output_size=2`
    >>> 
    >>> agent = get_agent(env.observation_space, 
                          env.action_space, 
                          {'hidden_sizes': [64, 64], 'activation': 'tanh'})
    >>> 
    >>> print(agent)

    Returns: 

    >>> MLP(
    >>>     (layers): Sequential(
    >>>         (0): Linear(in_features=4, out_features=64, bias=True)
    >>>         (1): ReLU()
    >>>         (2): Linear(in_features=64, out_features=64, bias=True)
    >>>         (3): ReLU()
    >>>         (4): Linear(in_features=64, out_features=2, bias=True)
    >>>     )
    >>> )

    :param obs_space: observation space of the environment
    :param action_space: action space of the environment
    :param kwargs: kwargs for the agent, see :py:mod:`rlib.agents` for more details.
    :param q_table: whether to use a QTable agent
    :param ddpg_q_agent: if True, the agent is a Q function and returns a scalar value for each state-action pair
    :param ppo_critic: if True, the agent is a critic and returns a scalar value for each state
    :type obs_space: gym.spaces
    :type action_space: gym.spaces
    :type kwargs: dict
    :type q_table: bool
    :type ddpg_q_agent: bool
    :type ppo_critic: bool

    :return: Either :class:`.MLP` or :class:`.QTable` depending on the parameters
    
    """

    if q_table:
        return QTable(**kwargs)

    if isinstance(action_space, Discrete):
        output_dim = action_space.n
    elif isinstance(action_space, MultiDiscrete):
        output_dim = action_space.nvec[0]
    elif isinstance(action_space, Box):
        n_dims = len(action_space.shape[1:])
        if n_dims != 1:
            raise ValueError("The action space should have 1 dimension.")
        output_dim = action_space.shape[1:][0]
    else:
        raise ValueError(f"Unknown action space {action_space}.")

    if isinstance(obs_space, Discrete):
        raise NotImplementedError("Discrete observation spaces are not supported yet. Consider using a QTable instead.")
    elif isinstance(obs_space, Box):
        n_dims = len(obs_space.shape[1:])
        if n_dims == 1:
            if ddpg_q_agent:
                kwargs["input_size"] = obs_space.shape[1:][0] + output_dim
                kwargs["output_size"] = 1
            elif ppo_critic:
                kwargs["input_size"] = obs_space.shape[1:][0]
                kwargs["output_size"] = 1
            else:
                kwargs["input_size"] = obs_space.shape[1:][0]
                kwargs["output_size"] = output_dim
            agent_type = "mlp"
        elif n_dims == 2:
            agent_type = "cnn"
        else:
            raise ValueError("The state space should have 1 or 2 dimensions.")
    else:
        raise ValueError(f"Unknown observation space {obs_space}.")


    if agent_type == "mlp":
        return MLP(**kwargs)
    elif agent_type == "cnn":
        raise NotImplementedError("CNNs are not supported yet.")
    else:
        raise ValueError(f"Unknown agent type {agent_type}.")
