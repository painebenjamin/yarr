import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FloatLayerNorm"]

class FloatLayerNorm(nn.LayerNorm):
    """
    A LayerNorm which always calculates in full precision (FP32).

    Will cast the output back to the input's dtype.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor, either FP32 or FP16
        :return: normalized input tensor in the same dtype as the input
        """
        return F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps
        ).type_as(x)
