import numpy as np
import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron (MLP) class.

    Note that, in order to use algorithm such that :class:`learning.evolution_strategy.EvolutionStrategy`,
    the gradient of the MLP should be disabled. This can be done by calling :func:`remove_grad` method,
    or by setting the `requires_grad` argument to False.

    Example:

    .. code-block:: python

        import torch
        from rlib.agents import MLP

        agent = MLP(4, [32, 32], 2, activation='relu', requires_grad=True)  # 4 observations, 2 actions, 2 hidden layers of 32 neurons each
        params = agent.get_params()  # get the parameters of the MLP
        agent.remove_grad()  # remove the gradient of the MLP
        agent.set_params(params)  # set the parameters of the MLP

        x = torch.randn(10, 4)  # 10 observations of size 4
        y = agent(x)  # forward pass of the MLP
        action = agent.get_action(x[0])  # get the actions to take


    :var layers: A torch.nn.Sequential object containing the layers of the MLP.
    :vartype layers: torch.nn.Sequential

    """

    def __init__(self, input_size, hidden_sizes, 
                 output_size, activation='relu', params=None, 
                 requires_grad=False, type_actions='discrete', action_space=None,
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
        :param params: The parameters of the MLP. If None, the parameters are initialized randomly. Default is None.
        :type params: dict, optional
        :param requires_grad: Whether to compute the gradient of the MLP. Default is False.
        :type requires_grad: bool, optional
        :raises ValueError: If `activation` is not one of 'relu', 'tanh' or 'sigmoid'.
        :param type_actions: The type of the actions. Should be one of 'discrete' or 'continuous'. Default is 'discrete'. If 'continuous', the output of the MLP is transformed to be in the range of the action space.
        :type type_actions: str, optional
        :param action_space: The action space of the environment. Default is None.
        :type action_space: gym.spaces.Box, optional

        """

        super().__init__()

        self.requires_grad = requires_grad
        self.type_actions = type_actions

        if self.type_actions == 'continuous':
            self.action_space = action_space

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


        if not requires_grad:
            self.remove_grad()

        if params is not None:
            self.set_params(params)

    def forward(self, x):
        """
        Computes the forward pass of the MLP.

        :param x: The input.
        :type x: torch.Tensor
        :return: The output of the MLP.
        :rtype: torch.Tensor
        """

        x = self.layers(x)

        # if the actions are continous, transform the output of the MLP to be in the range of the action space
        if self.type_actions == 'continuous':
            x = torch.tanh(x)
            x = (x + 1) / 2
            x = x * (torch.tensor(self.action_space.high) - torch.tensor(self.action_space.low)) + torch.tensor(self.action_space.low)

        return x

    def get_action(self, observation):
        """
        Given an observation from a `gymnasium.ENV`, returns the action to take.

        :param observation: The observation from the environment.
        :type observation: numpy.ndarray
        :return: The action to take.
        :rtype: int
        """
        observation = torch.tensor(observation).float()
        x = self(observation)
        if self.type_actions == 'discrete':
            return torch.argmax(x).numpy()
        else:
            return x.detach().numpy()
        

    def remove_grad(self):
        """
        Removes the gradient of the MLP.
        """

        for p in self.parameters():
            p.requires_grad = False
    
    def set_params(self, params):
        """
        Sets the parameters of the MLP.

        :param params: The parameters to set.
        :type params: dict
        """

        for k, p in self.named_parameters():
            p.copy_(params[k])
    
    def get_params(self):
        """
        Returns the parameters of the MLP.

        :return: The parameters of the MLP.
        :rtype: dict
        """

        return {k: p.clone() for k, p in self.named_parameters()}

    def init_layer_ppo(layer, std=np.sqrt(2), bias_constant=0.0):

        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_constant)

        return layer