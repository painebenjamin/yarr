# Released by Benjamin Paine under the Unlicense, see the LICENSE file or unlicense.org, 2024.

import torch
import torch.nn as nn

from typing import Any, Dict
from typing_extensions import Self

from ...utilities import summarize_module

__all__ = ["Module"]

class Module(nn.Module):
    """
    This class is a wrapper around nn.Module that provides some additional functionality.
    """
    @property
    def dtype(self) -> torch.dtype:
        """
        Get the dtype of the first parameter in the list of Module parameters.

        Note that this is unreliable in a mixed-precision context. In such cases, you should
        not use this and should instead write your own getters to ensure that you are getting
        the correct dtype from the correct module.

        :return: The dtype of the first parameter in the list of Module parameters.
        """
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        """
        Get the dtype of the first parameter in the list of Module parameters.

        This is /usually/ going to be reliable, but it is still possible for the device of the
        first parameter to be different from the device of the module itself. In such cases, you
        should not use this and should instead write your own getters to ensure that you are getting
        the correct device from the correct module.

        :return: The device of the first parameter in the list of Module parameters.
        """
        return next(self.parameters()).device

    def summarize(self) -> Dict[str, Any]:
        """
        Summarize the module by introspecting its attributes.

        :see: utilities.summarize_module
        :return: A dictionary containing the module's attributes.
        """
        return summarize_module(self)

    def best(self) -> Self:
        """
        Move the module to the best device and dtype.

        If no accelerators are available, the module will be left on the CPU.

        :return: The module itself.
        """
        if torch.cuda.is_available():
            return self.to(device="cuda")
        elif torch.backends.mps.is_available():
            return self.to(device="mps")
        return self
