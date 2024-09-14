# Released by Benjamin Paine under the Unlicense, see the LICENSE file or unlicense.org, 2024.

import torch
import torch.nn as nn

from typing import Optional

from ..modules import Module, AdaptiveModulator

__all__ = ["FinalLayer", "ModulatingFinalLayer"]

class FinalLayer(Module):
    """
    A simple final layer that applies a layer normalization and a linear layer.

    Output follows the formula:

        f(x) = W * norm(x) + b

    Where W is the weight matrix, b is the bias vector, and norm is the formula:

        norm(x) = (x - mean(x)) / sqrt(var(x) + epsilon)
    """
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        norm_epsilon: float = 1e-6,
        zero_init: bool = True,
    ) -> None:
        """
        :param hidden_size: The size of the hidden layer.
        :param output_size: The size of the output layer.
        :param norm_epsilon: The epsilon value for normalization.
        :param zero_init: Whether to initialize the weights to zero.
        """
        super(FinalLayer, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=norm_epsilon)
        self.fc = nn.Linear(hidden_size, output_size, bias=True)

        if zero_init:
            nn.init.constant_(self.fc.weight, 0)
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the final layer.

        :param x: The input tensor.
        :return: The output tensor.
        """
        x = self.norm(x)
        x = self.fc(x)
        return x

class ModulatingFinalLayer(FinalLayer):
    """
    A final layer that applies a layer normalization, a linear layer, and a modulator.

    Output follows the formula:

        f(x, y) = W * norm(x * (1 + y_0) + y_1) + b

    Where W is the weight matrix, b is the bias vector, y_0 is the shift modulation,
    y_1 is the scale modulation, and norm is the formula:

        norm(x) = (x - mean(x)) / sqrt(var(x) + epsilon)
    """
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        norm_epsilon: float = 1e-6,
        zero_init: bool = True,
    ) -> None:
        """
        :param hidden_size: The size of the hidden layer.
        :param output_size: The size of the output layer.
        :param norm_epsilon: The epsilon value for normalization.
        :param zero_init: Whether to initialize the weights to zero.
        """
        super(ModulatingFinalLayer, self).__init__(
            hidden_size=hidden_size,
            output_size=output_size,
            norm_epsilon=norm_epsilon,
            zero_init=zero_init,
        )
        self.modulator = AdaptiveModulator(
            hidden_size=self.hidden_size,
            num_modulations=2,
            modulate_bias=True
        )

    def forward(self, x: torch.Tensor, modulations: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the final layer.

        :param x: The input tensor.
        :param modulations: The modulations for the adaptive modulator.
        :return: The output tensor.
        """
        if modulations is None:
            return super(ModulatingFinalLayer, self).forward(x)

        shift, scale = self.modulator(modulations)

        x = self.norm(x)
        x = self.modulator.modulate(x, shift, scale)
        x = self.fc(x)
        return x
