# Released by Benjamin Paine under the Unlicense, see the LICENSE file or unlicense.org, 2024.

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from .base import Module

__all__ = ["Attention"]

class Attention(Module):
    """
    The attention mechanism is foundational to deep learning as
    a whole. In essence, it allows the model to focus on specific
    parts of the input sequence when making predictions.

    It is fundementally composed of two parts; *keys* and *values*,
    similar to a dictionary. The *keys* are the input sequence, and
    the *values* are the *weights* that are assigned to each key.

    A query is then passed to the attention mechanism, which is used
    to compute the weights that are assigned to each key. The weights
    are then used to compute the output.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        bias: bool = False,
    ) -> None:
        """
        :param dim: The dimension of the query tensor.
        :param num_heads: The number of attention heads (parallel attention mechanisms).
        :param bias: Whether to include a bias term in the linear transformation.
        """
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // self.num_heads
        self.inner_dim = self.head_dim * self.num_heads

        self.queries = nn.Linear(self.dim, self.inner_dim, bias=bias)
        self.keys = nn.Linear(self.dim, self.inner_dim, bias=bias)
        self.values = nn.Linear(self.dim, self.inner_dim, bias=bias)
        self.output = nn.Linear(self.inner_dim, self.dim, bias=bias)

        self.query_norm = nn.LayerNorm(self.inner_dim)
        self.key_norm = nn.LayerNorm(self.inner_dim)

    def reshape_for_broadcast(
        self,
        tensor: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        This helper function reshapes a tensor to a given shape, while
        preserving the number of elements in the tensor. This is useful
        for broadcasting operations.

        :param tensor: The tensor to reshape.
        :param target: The target tensor to reshape to.
        :return: The reshaped tensor
        """
        assert target.ndim > 1, "The target tensor must have at least two dimensions."
        tensor_view = tensor[: target.shape[1]]
        shape = [
            d if i == 1 or i == target.ndim - 1
            else 1
            for i, d in enumerate(target.shape)
        ]
        return tensor_view.view(*shape)

    def rotary_positional_encoding(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        frequencies: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the rotary positional encoding method (RoPE) to the
        query and key tensors. This method is used to inject positional
        information into the input sequence.

        :param query: The query tensor.
        :param key: The key tensor.
        :param frequencies: The frequencies tensor, encoded in cis 
            (cosine + i * sine) form.
        :return: The query and key tensors with positional encoding
            applied.
        """
        query_dtype = query.dtype
        query_complex = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
        query_frequencies = self.reshape_for_broadcast(frequencies, query_complex)
        query_out = torch.view_as_real(query_complex * query_frequencies).flatten(3)

        key_dtype = key.dtype
        key_complex = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
        key_frequencies = self.reshape_for_broadcast(frequencies, key_complex)
        key_out = torch.view_as_real(key_complex * key_frequencies).flatten(3)

        return query_out.to(query_dtype), key_out.to(key_dtype)

    def forward(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        The forward method of the attention mechanism.

        :param x: The input tensor.
        :param f: The frequencies tensor, encoded in cis (cosine + i * sine) form.
        :return: The output tensor.
        """
        b, s, _ = x.shape

        x_q, x_k, x_v = self.queries(x), self.keys(x), self.values(x)

        x_q = self.query_norm(x_q)
        x_k = self.key_norm(x_k)

        x_q = x_q.view(b, s, self.num_heads, self.head_dim)
        x_k = x_k.view(b, s, self.num_heads, self.head_dim)
        x_v = x_v.view(b, s, self.num_heads, self.head_dim)

        x_q, x_k = self.rotary_positional_encoding(x_q, x_k, f)

        x_q = x_q.permute(0, 2, 1, 3)
        x_k = x_k.permute(0, 2, 1, 3)
        x_v = x_v.permute(0, 2, 1, 3)

        output = F.scaled_dot_product_attention(
            x_q,
            x_k,
            x_v,
            dropout_p=0.0,
            is_causal=False
        )
        output = output.permute(0, 2, 1, 3)
        output = output.flatten(-2)
        output = self.output(output)

        return output
