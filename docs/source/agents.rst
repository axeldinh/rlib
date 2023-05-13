
Agents
======

This library contains the agents that are used to run the different algorithms.

Available Agents:
-----------------

 * :class:`QTable<agents.q_table.QTable>`
 * :class:`MLP<agents.mlp.MLP>`

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
    .. automethod:: get_params
    .. automethod:: set_params
    .. automethod:: get_action
    .. automethod:: remove_grad
