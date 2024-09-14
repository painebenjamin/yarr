# Released by Benjamin Paine under the Unlicense, see the LICENSE file or unlicense.org, 2024.

import torch

from typing import Tuple, List, Optional

from ..components import Module
from ..architectures import DiffusionTransformer

__all__ = ["RectifiedFlow"]

class RectifiedFlow(Module):
    """
    Rectified flow is a diffusion transformer model that connects image and noise in a straight line.
    Each sample is a linear interpolation between the input image and a noise sample like so:

        z_t = (1 - t) * x + t * z_1

    Where x is the input image, z_1 is a noise sample, and t is a scalar between 0 and 1. This seems
    obvious, but for a long time performed more poorly than other models. In 2024 stability AI published
    a paper showing the benefits of this approach coupled with perceptual noise scaling (arxiv:2403.03206).

    :see: https://arxiv.org/abs/2209.15571 [Building Normalizing Flows with Stochastic Interpolants, 2022]
    :see: https://arxiv.org/abs/2403.03206 [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis, 2024]
    """
    def __init__(
        self,
        in_channels: int=3,
        out_channels: int=3,
        input_size: int=32,
        patch_size: int=2,
        dim: int=512,
        num_layers: int=5,
        num_heads: int=16,
        multiple_of: int=256,
        norm_epsilon: float=1e-5,
        label_dropout: float=0.1,
        num_classes: int=10,
    ) -> None:
        """
        :param in_channels: The number of input channels, for example 1 for grayscale or 3 for RGB.
        :param out_channels: The number of output channels, for example 1 for grayscale or 3 for RGB.
        :param input_size: The size of the input image, for example 32 for 32x32.
        :param patch_size: The size of the patch, for example 2 for 2x2.
        :param dim: The dimension of the model, for example 512.
        :param num_layers: The number of layers, for example 5.
        :param num_heads: The number of heads, for example 16.
        :param multiple_of: The multiple of the model dimension, for example 256.
        :param norm_epsilon: The epsilon value for normalization, for example 1e-5.
        :param label_dropout: The dropout rate for the labels, for example 0.1.
        :param num_classes: The number of classes, for example 10 for MNIST or CIFAR-10, 100 for CIFAR-100, etc.
        """
        super().__init__()
        self.model = DiffusionTransformer(
            in_channels=in_channels,
            out_channels=out_channels,
            input_size=input_size,
            patch_size=patch_size,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            multiple_of=multiple_of,
            norm_epsilon=norm_epsilon,
            label_dropout=label_dropout,
            num_classes=num_classes,
        )

    @property
    def in_channels(self) -> int:
        """
        :return: The number of input channels.
        """
        return self.model.in_channels

    @property
    def out_channels(self) -> int:
        """
        :return: The number of output channels.
        """
        return self.model.out_channels

    @property
    def num_classes(self) -> int:
        """
        :return: The number of classes.
        """
        return self.model.num_classes

    @property
    def input_size(self) -> int:
        """
        :return: The input size.
        """
        return self.model.input_size

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        generator: Optional[torch.Generator]=None
    ) -> Tuple[
        torch.Tensor,
        List[Tuple[torch.Tensor, torch.Tensor]]
    ]:
        """
        Forward pass of the model.

        :param x: The input tensor.
        :param cond: The conditioning tensor.
        :return: The output tensor and the loss per class.
        """
        b = x.shape[0]
        t = torch.randn((b,), generator=generator).to(x.device)
        t = torch.sigmoid(t)
        t_exp = t.view([b, *([1] * len(x.shape[1:]))])
        z_1 = torch.randn(x.shape, generator=generator).to(x.device)
        z_t = (1 - t_exp) * x + t_exp * z_1
        v_theta = self.model(z_t, t, cond)
        batchwise_mse = ((z_1 - x - v_theta) ** 2).mean(dim=list(range(1, len(x.shape))))
        loss: torch.Tensor = batchwise_mse.mean()
        t_list = batchwise_mse.detach().cpu().reshape(-1).tolist()
        t_t_loss = [(t_v, t_loss) for t_v, t_loss in zip(t, t_list)]
        return loss, t_t_loss

    def sample(
        self,
        z: torch.Tensor,
        cond: torch.Tensor,
        sample_steps: int=50,
        cfg: float=2.0,
        generator: Optional[torch.Generator]=None
    ) -> List[torch.Tensor]:
        """
        Sample from the model.

        :param z: The input tensor.
        :param cond: The conditioning tensor.
        :param sample_steps: The number of steps to sample.
        :param cfg: The coefficient for classifier-free guidance.
        :return: A list of tensors showing the diffusion process.
        """
        b = z.shape[0]
        d_t = torch.tensor([1.0 / sample_steps] * b).to(z.device)
        d_t = d_t.view([b, *([1] * len(z.shape[1:]))])

        images = [z]
        for i in range(sample_steps, 0, -1):
            t = torch.tensor([i / sample_steps] * b).to(z.device)

            v_c = self.model(z, t, cond)
            if cfg > 0:
                v_u = self.model(z, t, torch.ones_like(cond) * 10)
                v_c = v_u + cfg * (v_c - v_u)

            z = z - d_t * v_c
            images.append(z)
        return images
