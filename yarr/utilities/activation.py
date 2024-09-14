# Released by Benjamin Paine under the Unlicense, see the LICENSE file or unlicense.org, 2024.

import torch.nn as nn

from typing import Optional, Any
from typing_extensions import Literal

__all__ = ["ActivationFunctionLiteral", "get_activation"]

ActivationFunctionLiteral = Literal["relu", "gelu", "silu", "swish", "mish", "tanh", "sigmoid", "identity"]

def get_activation(
    activation: Optional[ActivationFunctionLiteral],
    *args: Any,
    **kwargs: Any
) -> nn.Module:
    """
    Returns an activation function module based on the provided name.

    The supported activation functions are:

    - "relu": Rectified Linear Unit, commonly used in many neural networks.
    - "gelu": Gaussian Error Linear Unit, often used in transformer architectures.
    - "silu": Sigmoid Linear Unit, also known as "swish", a smooth non-linear activation.
    - "swish": Alias for "silu" for consistency with other naming conventions.
    - "mish": Another smooth activation function similar to "swish".
    - "tanh": Hyperbolic tangent, used in tasks requiring a bounded output in the range [-1, 1].
    - "sigmoid": Produces outputs in the range [0, 1], useful for binary classification tasks.
    - "identity": No-op activation, returns the input unchanged. This can be useful when you want
      to effectively "turn off" activation without modifying the model structure.

    :param activation: A string representing the name of the desired activation function.
                       If None is provided, or if "identity" is selected, no activation will be applied.
    :param args: Additional positional arguments to be passed to the activation function, if any.
    :param kwargs: Additional keyword arguments to be passed to the activation function, if any.
    :return: A PyTorch activation function module (`torch.nn.Module`).
    :raises ValueError: If an unknown activation function is passed.
    """
    activation_map = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "mish": nn.Mish,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "identity": nn.Identity,
        None: nn.Identity
    }.get(activation, None)
    if activation_map is None:
        raise ValueError(f"Activation function '{activation}' not found.")
    return activation_map(*args, **kwargs) # type: ignore[no-any-return]
