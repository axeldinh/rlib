from .q_table import QTable
from .mlp import MLP

def get_agent(agent_type, **kwargs):

    if agent_type == "mlp":
        return MLP(**kwargs)
    elif agent_type == "q_table":
        return QTable(**kwargs)
    else:
        raise ValueError(f"Unknown agent type {agent_type}.")