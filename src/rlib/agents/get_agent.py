from rlib.agents.q_table import QTable
from rlib.agents.mlp import MLP

from gymnasium.spaces import Discrete, Box

def get_agent(obs_space, action_space, kwargs, q_table=False):
    """ Global function to get an agent from its type and parameters

    This is the function used in the algorithms when a kwargs for an agent are given.
    The agent_type is usually automatically inferred from the environment's observation and ction spaces.

    :param agent_type: type of the agent, either "mlp" or "q_table", "cnn" is not supported yet
    :type agent_type: str

    :return: agent
    
    """

    if q_table:
        return QTable(**kwargs)

    if isinstance(action_space, Discrete):
        output_dim = action_space.n
    elif isinstance(action_space, Box):
        n_dims = len(action_space.shape)
        if n_dims != 1:
            raise ValueError("The action space should have 1 dimension.")
        output_dim = action_space.shape[0]
    else:
        raise ValueError(f"Unknown action space {action_space}.")

    if isinstance(obs_space, Discrete):
        raise NotImplementedError("Discrete observation spaces are not supported yet. Consider using a QTable instead.")
    elif isinstance(obs_space, Box):
        n_dims = len(obs_space.shape)
        if n_dims == 1:
            kwargs["input_size"] = obs_space.shape[0]
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
