# Released by Benjamin Paine under the Unlicense, see the LICENSE file or unlicense.org, 2024.

import torch
import torch.nn as nn

from torch.optim import Optimizer # type: ignore[attr-defined]

from typing import Any, Dict

__all__ = [
    "summarize_parameter",
    "summarize_module",
    "summarize_metadatum",
    "summarize_tensor",
    "summarize_tensors",
    "summarize_optimizer",
]

def summarize_optimizer(optimizer: Optimizer) -> Dict[str, Any]:
    """
    Summarize an optimizer by introspecting its attributes.

    :param optimizer: The optimizer to summarize.
    :return: A dictionary containing a summary of the optimizer
    """
    return {
        "name": optimizer.__class__.__name__,
        "metadata": {
            k: summarize_metadatum(v)
            for k, v in optimizer.__dict__.items()
            if not k.startswith("_")
            and k not in ["param_groups"]
        },
        "param_groups": [
            {
                "lr": group["lr"],
                "betas": group["betas"],
                "eps": group["eps"],
                "weight_decay": group["weight_decay"],
                "amsgrad": group["amsgrad"],
                "num_params": len(group["params"]),
            }
            for group in optimizer.param_groups
        ],
    }

def summarize_tensors(*tensors: torch.Tensor, **named_tensors: torch.Tensor) -> Dict[str, Any]:
    """
    Summarize tensors by introspecting their attributes.

    :param tensors: The tensors to summarize. Optional.
    :param named_tensors: The named tensors to summarize. Optional.
    :return: A dictionary containing a summary of the tensors.
    """
    return {
        **{f"tensor_{i}": summarize_tensor(tensor) for i, tensor in enumerate(tensors)},
        **{name: summarize_tensor(tensor) for name, tensor in named_tensors.items()},
    }

def summarize_tensor(tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Summarize a tensor by introspecting its attributes.

    :param tensor: The tensor to summarize.
    :return: A dictionary containing a summary of the tensor.
    """
    if tensor.is_complex():
        return {
            "shape": tuple(tensor.shape),
            "dtype": tensor.dtype,
            "numel": tensor.numel(),
            "requires_grad": tensor.requires_grad,
            "real": summarize_tensor(tensor.real),
            "imag": summarize_tensor(tensor.imag),
        }
    elif tensor.is_floating_point():
        return {
            "shape": tuple(tensor.shape),
            "dtype": tensor.dtype,
            "numel": tensor.numel(),
            "requires_grad": tensor.requires_grad,
            "min": tensor.min().item(),
            "max": tensor.max().item(),
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
        }
    else:
        return {
            "shape": tuple(tensor.shape),
            "dtype": tensor.dtype,
            "numel": tensor.numel(),
            "requires_grad": tensor.requires_grad,
            "min": tensor.min().item(),
            "max": tensor.max().item(),
        }

def summarize_parameter(parameter: nn.Parameter) -> Dict[str, Any]:
    """
    Summarize a parameter by introspecting its attributes.

    :param parameter: The parameter to summarize.
    :return: A dictionary containing a summary of the parameter.
    """
    first_parameter_value = parameter.data.flatten()[0].item()
    constant = torch.all(parameter.data == first_parameter_value).item()
    return {
        "shape": tuple(parameter.shape),
        "dtype": parameter.dtype,
        "numel": parameter.numel(),
        "requires_grad": parameter.requires_grad,
        "init": None if not constant else first_parameter_value,
    }

def summarize_metadatum(metadatum: Any) -> Dict[str, Any]:
    """
    Summarize a metadatum by introspecting its attributes.

    :param metadatum: The metadatum to summarize.
    :return: A dictionary containing a summary of the metadatum.
    """
    if isinstance(metadatum, torch.Tensor):
        return summarize_tensor(metadatum)
    elif isinstance(metadatum, nn.Parameter):
        return summarize_parameter(metadatum)
    else:
        return {"type": type(metadatum).__name__, "value": metadatum}

def summarize_module(module: nn.Module) -> Dict[str, Any]:
    """
    Summarize a module by introspecting its attributes.

    :param module: The module to summarize.
    :return: A dictionary containing a summary of the module
    """
    return {
        "name": module.__class__.__name__,
        "metadata": {k: summarize_metadatum(v) for k, v in module.__dict__.items() if not k.startswith("_")},
        "children": {k: summarize_module(v) for k, v in module.named_children()},
        "local_parameters": {k: summarize_parameter(v) for k, v in module.named_parameters(recurse=False)},
        "total_parameters": sum(1 for p in module.parameters()),
        "total_parameter_numel": sum(p.numel() for p in module.parameters()),
    }
