from .q_table import QTable
from .mlp import MLP

def get_agent(agent_type, **kwargs):
    """ Global function to get an agent from its type and parameters

    This is the function used in the algorithms when a kwargs for an agent are given.
    The agent_type is usually automatically inferred from the environment's observation and ction spaces.

    :param agent_type: type of the agent, either "mlp" or "q_table", "cnn" is not supported yet
    :type agent_type: str

    :return: agent
    
    """

    if agent_type == "mlp":
        return MLP(**kwargs)
    elif agent_type == "q_table":
        return QTable(**kwargs)
    else:
        raise ValueError(f"Unknown agent type {agent_type}.")