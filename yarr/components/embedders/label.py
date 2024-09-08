# Released by Benjamin Paine under the Unlicense, see the LICENSE file or unlicense.org, 2024.

import torch
import torch.nn as nn

from typing import Optional

from ..modules import Module

__all__ = ["LabelEmbedder"]

class LabelEmbedder(Module):
    """
    A simple label embedder that maps the labels to a low-dimensional space.
    """
    def __init__(
        self,
        num_classes: int,
        hidden_size: int,
        dropout: float = 0.0,
    ) -> None:
        """
        :param num_classes: The number of classes (i.e., the number of distinct labels).
        :param hidden_size: The size of the hidden layer.
        :param dropout: The dropout rate to use during training.
        """
        super(LabelEmbedder, self).__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.table = nn.Embedding(
            num_classes + (dropout > 0), # Add one for the dropout mask.
            hidden_size
        )

    def drop(
        self,
        labels: torch.Tensor,
        drop_ids: Optional[torch.Tensor]=None
    ) -> torch.Tensor:
        """
        Apply dropout to the labels.

        :param labels: The labels to apply dropout to.
        :param drop_ids: The indices of the labels to drop. If none, use random labels.
        """
        if (self.dropout > 0 and self.training) or drop_ids is not None:
            if drop_ids is None:
                drop_ids = torch.rand(labels.shape[0]) < self.dropout
                drop_ids = drop_ids.to(labels.device)
            else:
                drop_ids = drop_ids == 1
            labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(
        self,
        labels: torch.Tensor,
        drop_ids: Optional[torch.Tensor]=None
    ) -> torch.Tensor:
        """
        Embed the labels.

        :param labels: The labels to embed.
        :param drop_ids: The indices of the labels to drop. If none, use random labels.
        """
        labels = self.drop(labels, drop_ids)
        labels = self.table(labels)
        return labels
