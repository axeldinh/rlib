
Agents
======

This library contains the agents that are used to run the different algorithms.

Available Agents:
-----------------

 * :class:`QTable<agents.q_table.QTable>`
 * :class:`MLP<agents.mlp.MLP>`

A function is given to create an agent from a configuration dictionary:

.. autofunction:: rlib.agents.get_agent

This is useful for saving and loading agents automatically from file.

QTable
------

.. autoclass:: agents.q_table.QTable
        
    .. automethod:: __init__
    .. automethod:: get_action
    .. automethod:: discretize
    .. automethod:: sample
    .. automethod:: update
    

MLP
---

.. autoclass:: agents.mlp.MLP
    
    .. automethod:: __init__
