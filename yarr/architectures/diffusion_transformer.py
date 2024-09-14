# Released by Benjamin Paine under the Unlicense, see the LICENSE file or unlicense.org, 2024.

import torch
import torch.nn as nn

from ..components import (
    Module,
    ModulatingFinalLayer,
    TimestepEmbedder,
    LabelEmbedder,
    TransformerBlock,
)

__all__ = ["DiffusionTransformer"]

class DiffusionTransformer(Module):
    """
    The Diffusion Transformer model.

    For an explanation of transformers, see the TransformerBlock class.

    The easiest way to imagine the diffusion process is to imagine the reverse instead, where
    we start with a sample (e.g. an image) and then slowly degrade it by adding noise at each
    timestep, eventually resulting in pure noise. The diffusion process is the reverse of this,
    where we start with pure noise and then slowly refine it to produce a sample. This is a simplification
    of the actual process, but it is an easy way to think about it.

    :see: components.blocks.transformer.TransformerBlock
    :see: https://arxiv.org/abs/1503.03585 [Deep Unsupervised Learning using Nonequilibrium Thermodynamics, 2015]
    :see: https://arxiv.org/abs/1706.03762 [Attention is All You Need, 2017]
    :see: https://arxiv.org/abs/2010.11929 [An Image is Worth 16x16 Words, 2020]
    :see: https://arxiv.org/abs/2006.11239 [Denoising Diffusion Probabilistic Models, 2020]
    :see: https://arxiv.org/abs/2105.05233 [Diffusion Models Beat GANs on Image Synthesis, 2021]
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
        :param patch_size: The size of the patches, for example 2 for 2x2 patches.
        :param dim: The dimension of the model, for example 512.
        :param num_layers: The number of transformer layers, for example 5.
        :param num_heads: The number of attention heads, for example 16.
        :param multiple_of: The dimension must be a multiple of this number, for example 256.
        :param norm_epsilon: The epsilon value for normalization layers, for example 1e-5.
        :param label_dropout: The dropout rate for the label embedding, for example 0.1.
        :param num_classes: The number of classes in the dataset, for example 10 (MNIST, CIFAR-10),
                            100 (CIFAR-100) or 1000 (ImageNet).
        """
        super(DiffusionTransformer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.patch_size = patch_size
        self.dim = dim
        self.num_heads = num_heads
        self.num_classes = num_classes

        # Input convolutions
        self.input = nn.Sequential(
            nn.Conv2d(in_channels, dim // 2, kernel_size=5, padding=2, stride=1),
            nn.SiLU(),
            nn.GroupNorm(32, dim // 2),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=5, padding=2, stride=1),
            nn.SiLU(),
            nn.GroupNorm(32, dim // 2),
        )

        # The three embeddings
        self.x_embedder = nn.Linear(patch_size * patch_size * dim // 2, dim, bias=True)
        self.t_embedder = TimestepEmbedder(min(dim, 1024))
        self.y_embedder = LabelEmbedder(num_classes, min(dim, 1024), label_dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                multiple_of=multiple_of,
                norm_epsilon=norm_epsilon,
                layer_id=i
            )
            for i in range(num_layers)
        ])

        # Final layers
        self.final_layer = ModulatingFinalLayer(
            hidden_size=dim,
            output_size=patch_size * patch_size * out_channels,
        )

        # Initialization
        nn.init.constant_(self.x_embedder.bias, 0)
        self.frequencies = self.generate_frequencies()

    def generate_frequencies(self) -> torch.Tensor:
        """
        The cosine/sine frequencies for the positional encoding.

        :return: The frequencies.
        """
        dim = self.dim // self.num_heads
        t = torch.arange(0, 4096)
        exp = torch.arange(0, dim, 2)[: (dim // 2)].float() / dim

        frequencies = 1.0 / (10_000 ** exp)
        frequencies = torch.outer(t, frequencies).float()
        frequencies = torch.polar(torch.ones_like(frequencies), frequencies)
        frequencies = frequencies.to(self.device)

        return frequencies # type: ignore[no-any-return]

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Patchify the input tensor.

        :param x: The input tensor.
        :return: The patchified tensor.
        """
        b, c, h, w = x.shape
        x = x.view(
            b, c,
            h // self.patch_size, self.patch_size,
            w // self.patch_size, self.patch_size
        )
        x = x.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unpatchify the input tensor.

        :param x: The input tensor.
        :return: The unpatchified tensor.
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(x.shape[0], c, h * p, w * p)
        return x

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        :param x: The input tensor.
        :param t: The timestep tensor.
        :param y: The label tensor.
        """
        # Input convolutions
        x = self.input(x)

        # Patchify the input
        x = self.patchify(x)

        # Embeddings
        x = self.x_embedder(x)
        t = self.t_embedder(t)
        y = self.y_embedder(y)

        # Transformer layers
        self.frequencies = self.frequencies.to(x.device)
        m = t.to(x.dtype) + y.to(x.dtype)
        for layer in self.layers:
            x = layer(x, self.frequencies[: x.shape[1]], m)

        # Final layer
        x = self.final_layer(x, m)

        # Unpatchify the output
        x = self.unpatchify(x)
        return x
