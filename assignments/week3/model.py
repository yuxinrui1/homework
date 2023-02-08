import torch
from typing import Callable
from torch import nn


class MLP(nn.Module):
    """
    create Multiple layer perceptron
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.actv = activation()
        self.initializer = initializer

        for i in range(hidden_size):
            out_dim = hidden_size
            layer = nn.Linear(input_size, out_dim)
            input_size = out_dim
            self.initializer(layer.weight)
            self.layers += [layer]
            self.layers += [self.actv]
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch) -> (torch):
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        for layer in self.layers:
            x = self.actv(layer(x))
        return self.out(x)
