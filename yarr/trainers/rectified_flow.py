# Released by Benjamin Paine under the Unlicense, see the LICENSE file or unlicense.org, 2024.

import os
import click
import torch
import wandb
import numpy as np

from typing import Any, Optional, Dict, List
from math import sqrt
from PIL import Image

from torchvision.utils import make_grid # type: ignore[import-untyped]
from torchvision.transforms import ( # type: ignore[import-untyped]
    Pad,
    Normalize,
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
)

from .base import Trainer
from ..models import RectifiedFlow
from ..utilities import write_video

DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_NUM_EPOCHS = 100
DEFAULT_BATCH_SIZE = 250
DEFAULT_WORKERS = 4

__all__ = ["RectifiedFlowTrainer"]

class RectifiedFlowTrainer(Trainer):
    """
    Trainer for the Rectified Flow model.
    """
    num_validation_samples = 100

    @property
    def num_classes(self) -> int:
        """
        :return: The number of classes in the dataset.
        """
        return int(self.model.num_classes)

    @property
    def cifar_10(self) -> bool:
        """
        :return: Whether the dataset is CIFAR-10.
        """
        return self.project_name == "rectified-flow-cifar10"

    @property
    def cifar_100(self) -> bool:
        """
        :return: Whether the dataset is CIFAR-100.
        """
        return self.project_name == "rectified-flow-cifar100"

    @property
    def imagenet_1k_32(self) -> bool:
        """
        :return: Whether the dataset is ImageNet-1k 32x32.
        """
        return self.project_name == "rectified-flow-imagenet-1k-32"

    def transform(self, datum: torch.Tensor) -> torch.Tensor:
        """
        Transform a single datum (i.e. prepare it for the model).

        :param datum: The datum to transform.
        :return: The transformed datum.
        """
        if not hasattr(self, "_transform"):
            if self.cifar_10 or self.cifar_100 or self.imagenet_1k_32:
                self._transform = Compose([
                    RandomCrop(32),
                    RandomHorizontalFlip(),
                    Normalize(mean=(0.5,), std=(0.5,))
                ])
            else:
                self._transform = Compose([
                    Pad(2),
                    Normalize(mean=(0.5,), std=(0.5,))
                ])
        datum = datum.to(torch.float32) / 255.
        return self._transform(datum) # type: ignore[no-any-return]

    def loss(self, datum: Any) -> torch.Tensor:
        """
        Compute the loss for a single datum.

        :param datum: The datum to compute the loss for.
        :return: The loss for the datum.
        """
        try:
            if self.cifar_100:
                x = datum["img"]
                y = datum["fine_label"]
            elif self.cifar_10:
                x = datum["img"]
                y = datum["label"]
            else:
                x = datum["image"]
                y = datum["label"]

            x = self.transform(x)

            loss, t_t_loss = self.model(x, y)

            if not hasattr(self, "loss_bin"):
                self.loss_bin = {i: 0 for i in range(self.num_classes)}
            if not hasattr(self, "loss_count"):
                self.loss_count = {i: 1e-6 for i in range(self.num_classes)}

            for t, l in t_t_loss:
                self.loss_bin[int(t * self.num_classes)] += l
                self.loss_count[int(t * self.num_classes)] += 1

            return loss # type: ignore[no-any-return]
        except (KeyError, AttributeError) as e:
            raise ValueError(f"Unexpected datum format: {e}\n{datum}")

    @torch.no_grad()
    def evaluate(self) -> None:
        """
        Evaluate the model on the validation dataset.
        """
        # Log loss bins to weights and biases
        if self.use_wandb:
            wandb.log({
                f"loss_bin_{i}": self.loss_bin[i] / self.loss_count[i]
                for i in range(self.num_classes)
            })

        # Reset loss bins
        self.loss_bin = {i: 0 for i in range(self.num_classes)}
        self.loss_count = {i: 1e-6 for i in range(self.num_classes)}

        # Make samples
        generator = torch.Generator()
        generator.manual_seed(42)
        noise = torch.randn(
            (self.num_validation_samples, self.model.in_channels, self.model.input_size, self.model.input_size),
            generator=generator
        )

        if self.imagenet_1k_32:
            conds = [
                torch.arange(self.num_validation_samples * i, self.num_validation_samples * (i + 1)) % self.model.num_classes
                for i in range(10)
            ]
        else:
            conds = [torch.arange(0, self.num_validation_samples) % self.model.num_classes]

        for i, cond in enumerate(conds):
            images = self.model.sample(noise.to(self.device), cond.to(self.device))

            frames = []
            for image in images:
                # denormalize
                image = (image * 0.5) + 0.5
                image = image.clamp(0, 1)
                image = make_grid(image, nrow=int(sqrt(self.num_validation_samples)))
                image = image.permute(1, 2, 0).detach().cpu().numpy()
                image = (image * 255).astype(np.uint8)
                frames.append(Image.fromarray(image))

            if not hasattr(self, "final_frames"):
                self.final_frames: Dict[int, List[Image.Image]] = {}

            if i not in self.final_frames:
                self.final_frames[i] = []

            # Save the sampling images as an mp4
            write_video(
                frames + [frames[-1]] * 20, # hold the last frame for 2 sec
                os.path.join(self.output_dir, f"sampling_{i}.mp4"),
                fps=10
            )
            # Save the completed samples as an mp4
            self.final_frames[i].append(frames[-1])
            write_video(
                self.final_frames[i] + [self.final_frames[i][-1]] * 20, # hold the last frame for 2 sec
                os.path.join(self.output_dir, f"samples_{i}.mp4"),
                fps=10
            )
            # Log final frame
            if self.use_wandb:
                wandb.log({f"samples_{i}": wandb.Image(self.final_frames[i][-1])})

@click.command()
@click.option("-lr", "--learning-rate", default=DEFAULT_LEARNING_RATE, help="Learning rate for the optimizer.", show_default=True)
@click.option("-e", "--num-epochs", default=DEFAULT_NUM_EPOCHS, help="Number of epochs to train the model.", show_default=True)
@click.option("-w", "--num-workers", default=DEFAULT_WORKERS, help="Number of workers for the data loader.", show_default=True)
@click.option("-b", "--batch-size", default=DEFAULT_BATCH_SIZE, help="Batch size for training.", show_default=True)
@click.option("--evaluate-nth-batch", type=int, default=None, help="Evaluate the model every nth batch as opposed to only between epochs.", show_default=True)
@click.option("--wandb-entity", default=None, help="Weights and Biases entity.", type=str)
@click.option("--resume", is_flag=True, help="Resume training from the latest checkpoint.")
@click.option("--fashion-mnist", is_flag=True, help="Use Fashion MNIST dataset instead of MNIST.")
@click.option("--cifar-10", is_flag=True, help="Use CIFAR-10 dataset instead of MNIST.")
@click.option("--cifar-100", is_flag=True, help="Use CIFAR-100 dataset instead of MNIST.")
@click.option("--imagenet-1k-32", is_flag=True, help="Use ImageNet-1k 32x32 dataset instead of MNIST.")
def main(
    learning_rate: float=DEFAULT_LEARNING_RATE,
    num_epochs: int=DEFAULT_NUM_EPOCHS,
    num_workers: int=DEFAULT_WORKERS,
    batch_size: int=DEFAULT_BATCH_SIZE,
    evaluate_nth_batch: Optional[int]=None,
    wandb_entity:Optional[str]=None,
    resume: bool=False,
    fashion_mnist: bool=False,
    cifar_10: bool=False,
    cifar_100: bool=False,
    imagenet_1k_32: bool=False,
) -> None:
    """
    Train a Rectified Flow model on either MNIST, Fashion MNIST,
    CIFAR-10, CIFAR-100, or ImageNet-1k 32x32 datasets.
    """
    large = False
    extra_large = False
    if imagenet_1k_32:
        if cifar_10 or cifar_100 or fashion_mnist:
            print("Using ImageNet-1k 32x32 dataset, --cifar-10, --cifar-100, and --fashion-mnist flags will be ignored.")
        large = True
        extra_large = True
        cifar_10 = False
        cifar_100 = False
        fashion_mnist = False
    if cifar_100:
        if cifar_10 or fashion_mnist:
            print("Using CIFAR-100 dataset, --cifar-10 and --fashion-mnist flags will be ignored.")
        large = True
        cifar_10 = False
        fashion_mnist = False
    elif cifar_10:
        if fashion_mnist:
            print("Using CIFAR-10 dataset, --fashion-mnist flag will be ignored.")
        large = True
        cifar_100 = False
        fashion_mnist = False

    model = RectifiedFlow(
        in_channels=3 if large else 1,
        out_channels=3 if large else 1,
        dim=512 if extra_large else 256 if large else 64,
        num_layers=12 if extra_large else 10 if large else 6,
        num_heads=8 if large else 4,
        input_size=32,
        num_classes=1000 if imagenet_1k_32 else 100 if cifar_100 else 10,
        patch_size=2,
    )

    if imagenet_1k_32:
        training_dataset = "benjamin-paine/imagenet-1k-32x32"
        project_name = "rectified-flow-imagenet-1k-32"
    elif cifar_100:
        training_dataset = "uoft-cs/cifar100"
        project_name = "rectified-flow-cifar100"
    elif cifar_10:
        training_dataset = "uoft-cs/cifar10"
        project_name = "rectified-flow-cifar10"
    elif fashion_mnist:
        training_dataset = "zalando-datasets/fashion_mnist"
        project_name = "rectified-flow-fashion-mnist"
    else:
        training_dataset = "ylecun/mnist"
        project_name = "rectified-flow-mnist"

    trainer = RectifiedFlowTrainer(
        model=model,
        learning_rate=learning_rate,
        project_name=project_name,
        wandb_entity=wandb_entity,
        training_dataset=training_dataset,
        training_dataset_batch_size=batch_size,
        training_dataset_workers=num_workers,
    )

    if resume:
        trainer.resume()

    trainer(
        num_epochs=num_epochs,
        evaluate_nth_batch=evaluate_nth_batch,
    )

if __name__ == "__main__":
    main()
