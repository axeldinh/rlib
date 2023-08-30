import numpy as np
import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron (MLP) class.

    Note that, in order to use algorithm such that :class:`learning.evolution_strategy.EvolutionStrategy`,
    the gradient of the MLP should be disabled. This can be done by setting the `requires_grad` argument of each tensor to False.

    Example:

    .. code-block:: python
    
        import torch
        from rlib.agents import MLP

        agent = MLP(4, [32, 32], 2, activation='relu')  # 4 observations, 2 actions, 2 hidden layers of 32 neurons each
        x = torch.randn(4) 
        y = agent(x)

    :var layers: A torch.nn.Sequential object containing the layers of the MLP.
    :vartype layers: torch.nn.Sequential

    """

    def __init__(self, input_size, hidden_sizes, 
                 output_size, activation='relu', 
                 init_weights=None):
        """
        Initialize the MLP.

        :param input_size: The size of the input.
        :type input_size: int
        :param hidden_sizes: A list containing the sizes of the hidden layers.
        :type hidden_sizes: list
        :param output_size: The size of the output.
        :type output_size: int
        :param activation: The activation function to use. Should be one of 'relu', 'tanh' or 'sigmoid'. Default is 'relu'.
        :type activation: str, optional
        :raises ValueError: If `activation` is not one of 'relu', 'tanh' or 'sigmoid'.

        """

        super().__init__()


        if activation == 'relu':
            activation = nn.ReLU
        elif activation == 'tanh':
            activation = nn.Tanh
        elif activation == 'sigmoid':
            activation = nn.Sigmoid
        else:
            raise ValueError("activation should be one of 'relu', 'tanh' or 'sigmoid'")


        if hidden_sizes.__len__() == 0:
            self.layers = nn.Linear(input_size, output_size)
        else:
            self.layers = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]), activation())
            for i in range(0, len(hidden_sizes)-1):
                layer = nn.Linear(hidden_sizes[i], hidden_sizes[i+1])
                self.layers.append(MLP.init_layer_ppo(layer))
                self.layers.append(activation())
            last_layer = nn.Linear(hidden_sizes[-1], output_size)
            if init_weights == 'ppo_actor':
                self.layers.append(MLP.init_layer_ppo(last_layer, std=0.01))
            elif init_weights == 'ppo_critic':
                self.layers.append(MLP.init_layer_ppo(last_layer, std=1))
            else:
                self.layers.append(last_layer)

    def forward(self, x):
        """
        Computes the forward pass of the MLP.

        :param x: The input.
        :type x: torch.Tensor
        :return: The output of the MLP.
        :rtype: torch.Tensor
        """

        x = self.layers(x)

        return x

    def init_layer_ppo(layer, std=np.sqrt(2), bias_constant=0.0):

        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_constant)

        return layer