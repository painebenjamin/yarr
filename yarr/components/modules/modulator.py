# Released by Benjamin Paine under the Unlicense, see the LICENSE file or unlicense.org, 2024.

import torch
import torch.nn as nn

from typing import Tuple

from .base import Module

__all__ = ["AdaptiveModulator"]

class AdaptiveModulator(Module):
    """
    A class that allows for adaptive modulation of the input tensor.

    The number of modulations to to learn is specified by the `num_modulations` parameter.
    For example, a layer might want to have a shift and scale modulation, so `num_modulations` would be 2.

    Larger models may want to have 3 modulations (i.e. include a gating mechanism), and may want to
    learn modulations for more than one layer at a time, such as the transformer model which wants 6
    modulations (gate, shift and scale for the attention and feedforward layers).
    """
    def __init__(
        self,
        hidden_size: int,
        cond_size: int = 1024,
        num_modulations: int = 2,
        modulate_bias: bool = True
    ) -> None:
        """
        :param hidden_size: The size of the hidden layer.
        :param num_modulations: The number of modulations to apply.
        :param modulate_bias: Whether to include a bias term in the modulator.
        """
        super(AdaptiveModulator, self).__init__()
        self.hidden_size = hidden_size
        self.num_modulations = num_modulations
        self.adaptive_modulator = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                min(cond_size, hidden_size),
                num_modulations * hidden_size,
                bias=modulate_bias
            )
        )

    def modulate(
        self,
        input: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor
    ) -> torch.Tensor:
        """
        Modulate the input tensor.

        :param input: The input tensor to modulate.
        :param shift: The shift tensor.
        :param scale: The scale tensor.
        :return: The modulated tensor.
        """
        return input * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Get modulations for the input tensor.

        :param x: The input tensor.
        :return: The modulatation tensors
        :see self.modulate:
        """
        x = self.adaptive_modulator(x)
        return x.chunk(self.num_modulations, dim=1)
