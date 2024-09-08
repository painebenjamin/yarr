# Released by Benjamin Paine under the Unlicense, see the LICENSE file or unlicense.org, 2024.

import torch
import torch.nn as nn

from math import log

from ..modules import Module

__all__ = ["TimestepEmbedder"]

class TimestepEmbedder(Module):
    """
    A module that embeds timesteps.

    Timesteps are embedded as a concatenation of sinusoidal functions of different frequencies
    and phases. The frequencies are determined by the maximum period, and the phases are learned
    by a perceptron.

    To put it more simply, a given time `t` is encoded at varying degrees of accuracy using a
    combination of sine and cosine functions with different frequencies and phases, rather than
    trying to encode it as an integer or some other discrete value. This allows the model to
    learn better representations of time on multiple scales, i.e., it can perceive something that
    occurs every 10 steps as well as something that occurs every 100 steps.
    """
    def __init__(
        self,
        hidden_size: int,
        embed_size: int=256,
        max_period: int=10_000,
    ) -> None:
        """
        :param hidden_size: The hidden size.
        :param embed_size: The embedding size.
        :param max_period: The maximum period for the sinusoidal functions.
        """
        super(TimestepEmbedder, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.max_period = max_period
        self.perceptron = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @property
    def half_embed_size(self) -> int:
        """
        :return: Half of the embedding size.
        """
        return self.embed_size // 2

    def embed_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Get the embedding for the given timesteps.
        
        :param timesteps: The timesteps to embed.
        :return: The embedded timesteps.
        """
        frequencies = torch.exp(
            -log(self.max_period) * torch.arange(0, self.half_embed_size) / self.half_embed_size
        ).to(timesteps.device)
        timesteps = timesteps[:, None] * frequencies[None]
        timesteps = torch.cat([
            torch.cos(timesteps),
            torch.sin(timesteps),
        ], dim=-1)
        if self.embed_size % 2 != 0:
            timesteps = torch.cat([timesteps, torch.zeros_like(timesteps[:, :1])], dim=-1)
        return timesteps

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Embed the given timestep.

        :param timestep: The timestep to embed.
        :return: The embedded timestep.
        """
        timestep = self.embed_timesteps(timestep).to(self.dtype)
        timestep = self.perceptron(timestep)
        return timestep
