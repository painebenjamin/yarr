import torch
import torch.nn as nn

from typing import Optional

from ...utilities import (
    ActivationFunctionLiteral,
    get_normalized_dim,
    get_activation,
)

__all__ = [
    "MultiLayerPerceptron",
    "GatedMultiLayerPerceptron",
]

class MultiLayerPerceptron(nn.Module):
    """
    A simple multi-layer perceptron model that operates in a linear fashion.

    Output follows the formula:

       f(x) = W2 * g(W1 * x + b1) + b2 # With bias
       f(x) = W2 * g(W1 * x) # Without bias

    Where W1 and W2 are the weights of the hidden and output layers,
    b1 and b2 are the biases, and g is the activation function.

    These work well as linear classifiers on their own, but are generally used as
    part of a larger model, such as a convolutional neural network.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int]=None,
        output_dim: Optional[int]=None,
        activation: Optional[ActivationFunctionLiteral]="silu",
        multiple_of: int=8,
        bias: bool=True,
    ) -> None:
        """
        :param input_dim: The dimension of the input data.
        :param hidden_dim: The dimension of the hidden layer. Optional, default is 4 times the input dimension.
        :param output_dim: The dimension of the output data. Optional, default is the input dimension.
        :param activation: The activation function to use. Optional, default is "silu".
        :param multiple_of: The multiple to use for the hidden layer dimension. Optional, default is 8.
        :param bias: Whether to use bias in the linear layers. Optional, default is True.
        """
        super(MultiLayerPerceptron, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim * 4
        if output_dim is None:
            output_dim = input_dim

        self.input_dim = input_dim
        self.hidden_dim = get_normalized_dim(hidden_dim, multiple_of)
        self.output_dim = output_dim

        self.hidden = nn.Linear(self.input_dim, self.hidden_dim, bias=bias)
        self.activation = get_activation(activation)
        self.output = nn.Linear(self.hidden_dim, self.output_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        :param x: The input data.
        :return: The output data.
        """
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        return x

class GatedMultiLayerPerceptron(MultiLayerPerceptron):
    """
    An extension of the Multi-Layer perceptron with a parallel gating mechanism.

    Output follows the formula:

       f(x) = W3 * g(W1 * x + b1) * (W2 * x + b2) + b3 # With bias
       f(x) = W3 * W2 * g(W1 * x) * x # Without bias

    Where W1, W2 and W3 are the weights of the hidden, gate, and output layers,
    b1, b2 and b3 are their respective biases, and g is the activation function.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int]=None,
        output_dim: Optional[int]=None,
        activation: Optional[ActivationFunctionLiteral]="silu",
        multiple_of: int=8,
        bias: bool=True,
    ) -> None:
        """
        :param input_dim: The dimension of the input data.
        :param hidden_dim: The dimension of the hidden layer. Optional, default is 4 times the input dimension.
        :param output_dim: The dimension of the output data. Optional, default is the input dimension.
        :param activation: The activation function to use. Optional, default is "silu".
        :param multiple_of: The multiple to use for the hidden layer dimension. Optional, default is 8.
        :param bias: Whether to use bias in the linear layers. Optional, default is True.
        """
        super(GatedMultiLayerPerceptron, self).__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            activation=activation,
            multiple_of=multiple_of,
            bias=bias,
        )
        self.gate = nn.Linear(self.input_dim, self.hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        :param x: The input data.
        :return: The output data.
        """
        x = self.activation(self.hidden(x)) * self.gate(x)
        x = self.output(x)
        return x
