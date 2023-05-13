
Learning
========

This package contains the implementations of the learning algorithms.

Algorithms Available
--------------------


    * :class:`BaseAlgorithm<learning.base_algorithm.BaseAlgorithm>`
    * :class:`EvolutionStrategy<learning.evolution_strategy.EvolutionStrategy>`
    * :class:`QLearning<learning.q_learning.QLearning>`

Usage
-----

All the algorithms are implemented as classes, which can be used as follows:

.. code-block:: python

    import gymnasium
    from learning import Algorithm
    from agents import Agent

    env_fn = lambda render_mode=None: gymnasium.make('CartPole-v0', render_mode=render_mode, **kwargs)
    agent_fn = lambda params=None Agent(params=params, **kwargs)
    algorithm = Algorithm(env, agent, **kwargs)

    algorithm.train()
    algorithm.test()

----------------

BaseAlgorithm
-------------

The :class:`learning.base_algorithm.BaseAlgorithm` class gives the baselines for an algorithm to be useable
by our implementations:

.. autoclass:: learning.base_algorithm.BaseAlgorithm

    .. automethod:: learning.base_algorithm.BaseAlgorithm.__init__
    .. automethod:: learning.base_algorithm.BaseAlgorithm.train
    .. automethod:: learning.base_algorithm.BaseAlgorithm.test
    .. automethod:: learning.base_algorithm.BaseAlgorithm.save
    .. automethod:: learning.base_algorithm.BaseAlgorithm.load
    .. automethod:: learning.base_algorithm.BaseAlgorithm.save_plots
    .. automethod:: learning.base_algorithm.BaseAlgorithm.save_videos


EvolutionStrategy
-----------------

.. autoclass:: learning.evolution_strategy.EvolutionStrategy

    .. automethod:: learning.evolution_strategy.EvolutionStrategy.__init__
    .. automethod:: learning.evolution_strategy.EvolutionStrategy._get_random_parameters
    .. automethod:: learning.evolution_strategy.EvolutionStrategy._parameters_update
    .. automethod:: learning.evolution_strategy.EvolutionStrategy._get_test_parameters


QLearning
---------

.. autoclass:: learning.q_learning.QLearning
