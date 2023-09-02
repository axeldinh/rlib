
Learning
========

This package contains the implementations of the learning algorithms.

Algorithms Available
--------------------


    * :class:`BaseAlgorithm<learning.base_algorithm.BaseAlgorithm>`
    * :class:`EvolutionStrategy<learning.evolution_strategy.EvolutionStrategy>`
    * :class:`Q-Learning<learning.q_learning.QLearning>`
    * :class:`Deep Q-Learning<learning.deep_q_learning.DeepQLearning>`
    * :class:`Deep Deterministic Policy Gradient<learning.ddpg.DDPG>`

Usage
-----

All the algorithms are implemented as classes, which can be used as follows:

.. code-block:: python

    from learning import Algorithm
    from agents import Agent

    env_kwargs = {'id': 'CartPole-v1'}
    agent_kwargs = {'hidden_sizes': [64, 64]}
    algorithm = Algorithm(env_kwargs, agent_kwargs)

    algorithm.train()
    mean, std = algorithm.test()
    algorithm.save_plots()
    algorithm.save_videos()

----------------

BaseAlgorithm
-------------

The :class:`learning.base_algorithm.BaseAlgorithm` class gives the baselines for an algorithm to be useable
by our implementations.

For an algorithm to be useable, it must implement the following methods:

    * :meth:`learning.base_algorithm.BaseAlgorithm.train_`
    * :meth:`learning.base_algorithm.BaseAlgorithm.save`
    * :meth:`learning.base_algorithm.BaseAlgorithm.load`

.. autoclass:: learning.base_algorithm.BaseAlgorithm

    .. automethod:: learning.base_algorithm.BaseAlgorithm.__init__
    .. automethod:: learning.base_algorithm.BaseAlgorithm.train
    .. automethod:: learning.base_algorithm.BaseAlgorithm.train_
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

    .. automethod:: learning.q_learning.QLearning.__init__


Deep Q-Learning
---------------

.. autoclass:: learning.deep_q_learning.DeepQLearning

    .. automethod:: learning.deep_q_learning.DeepQLearning.__init__
    .. automethod:: learning.deep_q_learning.DeepQLearning._populate_replay_buffer
    .. automethod:: learning.deep_q_learning.DeepQLearning.update_weights

Deep Deterministic Policy Gradient
----------------------------------

.. autoclass:: learning.ddpg.DDPG

    .. automethod:: learning.ddpg.DDPG.__init__
    .. automethod:: learning.ddpg.DDPG._update_target_weights
    .. automethod:: learning.ddpg.DDPG._populate_replay_buffer
    .. automethod:: learning.ddpg.DDPG.update_weights
