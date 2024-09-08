# Released by Benjamin Paine under the Unlicense, see the LICENSE file or unlicense.org, 2024.

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from .base import Module

__all__ = ["FeedForward"]

class FeedForward(Module):
    """
    A simple feed forward neural network with 1 hidden layer.

    These are used to project lower-dimensional tensors to
    higher-dimensional ones, typically after an attention layer.
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int, # e.g. 64, 128, 256
        ffn_dim_multiplier: Optional[int] = None,
        bias: bool = False
    ) -> None:
        """
        :param dim: the dimension of the input tensor
        :param hidden_dim: the dimension of the hidden layer
        :param multiple_of: a number that `dim` must be a multiple of
        :param ffn_dim_multiplier: a multiplier for the hidden layer dimension
        :param bias: whether to include a bias term in the linear layers
        """
        super(FeedForward, self).__init__()
        self.hidden_dim = int(2 * hidden_dim / 3)
        if (ffn_dim_multiplier is not None):
            self.hidden_dim = ffn_dim_multiplier * self.hidden_dim
        self.hidden_dim = multiple_of * ((self.hidden_dim + multiple_of - 1) // multiple_of)

        self.input_layer_1 = nn.Linear(dim, self.hidden_dim, bias=bias)
        self.input_layer_2 = nn.Linear(dim, self.hidden_dim, bias=bias)
        self.output_layer = nn.Linear(self.hidden_dim, dim, bias=bias)

    def gate(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Performs SiLU gating on the input tensors.

        :param x1: a tensor of shape (batch_size, seq_len, dim)
        :param x2: a tensor of shape (batch_size, seq_len, dim)
        """
        return F.silu(x1) * x2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: a tensor of shape (batch_size, seq_len, dim)
        :return: a tensor of shape (batch_size, seq_len, dim)
        """
        x = self.gate(self.input_layer_1(x), self.input_layer_2(x))
        x = self.output_layer(x)
        return x
