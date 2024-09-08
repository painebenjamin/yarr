# Released by Benjamin Paine under the Unlicense, see the LICENSE file or unlicense.org, 2024.

import os
import wandb
import torch
import torch.optim as optim

from typing import List, Tuple, Union, Any, Optional, Dict

from tqdm import tqdm
from math import floor
from datasets import Dataset, load_dataset, interleave_datasets # type: ignore[import-untyped]
from torch.utils.data import DataLoader

from ..components import Module
from ..utilities import summarize_optimizer, summarize_module

def get_split_from_dataset_name(dataset_name: str) -> Tuple[str, str]:
    """
    Allows for passing a split name in the URL to a dataset,
    which is not supported by the Hugging Face Datasets library.

    :param dataset_name: The name of the dataset, optionally with a split name
    :return: A tuple with the dataset name and split name
    """
    name_split = dataset_name.split(":")
    if len(name_split) == 1:
        return name_split[0], "train"
    return name_split[0], name_split[1]

class Trainer:
    """
    A class for training a model on one or more datasets.
    """
    def __init__(
        self,
        model: Module,
        training_dataset: Union[str, List[str]],
        training_dataset_streaming: bool=False,
        training_dataset_batch_size: int=256,
        training_dataset_workers: int=4,
        output_dir: str=os.path.join(os.getcwd(), "output"),
        project_name: Optional[str]=None, # For Weights & Biases
        wandb_entity: Optional[str]=None, # For Weights & Biases
        learning_rate: float=1e-3,
    ) -> None:
        """
        :param model: The model to train
        :param training_dataset: The dataset to train on
        :param training_dataset_streaming: Whether to stream the dataset
        :param training_dataset_batch_size: The batch size for training
        :param training_dataset_workers: The number of workers for the training dataset
        :param output_dir: The directory to save the model and samples to.
        :param project_name: The name of the Weights & Biases project
        :param wandb_entity: The entity for the Weights & Biases project
        :param learning_rate: The learning rate for the optimizer
        """
        self.model = model
        self.learning_rate = learning_rate
        self.training_dataset = training_dataset
        self.training_dataset_batch_size = training_dataset_batch_size
        self.training_dataset_streaming = training_dataset_streaming
        self.training_dataset_workers = training_dataset_workers
        self.output_dir = output_dir
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate) # type: ignore[attr-defined]
        if project_name is not None:
            self.output_dir = os.path.join(self.output_dir, project_name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.project_name = project_name
        self.wandb_entity = wandb_entity
        self.total_epochs = 0

    @property
    def device(self) -> torch.device:
        """
        :return: The device the model is on
        """
        return self.model.device

    @property
    def training(self) -> Dataset:
        """
        :return: The training dataset, loaded and formatted for PyTorch
        """
        if not hasattr(self, "_training_dataset"):
            if isinstance(self.training_dataset, str):
                training_dataset_paths = [self.training_dataset]
            else:
                training_dataset_paths = self.training_dataset
            training_datasets = [
                load_dataset(repo_id, split=split, streaming=self.training_dataset_streaming)
                for repo_id, split in [
                    get_split_from_dataset_name(dataset_name)
                    for dataset_name in training_dataset_paths
                ]
            ]
            if len(training_datasets) == 1:
                self._training_dataset = training_datasets[0]
            else:
                self._training_dataset = interleave_datasets(training_datasets)
            self._training_dataset = self._training_dataset.with_format("torch")
        return self._training_dataset

    @property
    def dataloader(self) -> DataLoader[Dict[str, Any]]:
        """
        :return: The DataLoader for the training dataset
        """
        if not hasattr(self, "_dataloader"):
            self._dataloader = DataLoader(
                self.training,
                batch_size=self.training_dataset_batch_size,
                shuffle=True,
                num_workers=self.training_dataset_workers,
                drop_last=self.training_dataset_batch_size > 1,
            )
        return self._dataloader

    @property
    def use_wandb(self) -> bool:
        """
        :return: Whether to use Weights & Biases
        """
        return self.project_name is not None and self.wandb_entity is not None

    def loss(self, datum: Any) -> torch.Tensor:
        """
        Calculate the loss for the model.

        :param datum: The data to calculate the loss for
        :return: The loss
        """
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, epoch: int=0) -> None:
        """
        Evaluate the model.

        :param epoch: The epoch number at the time of evaluation
        """
        raise NotImplementedError

    def align(self, datum: Any) -> Any:
        """
        Align the data to the model's input.

        :param datum: The data to align
        :return: The aligned data
        """
        if isinstance(datum, dict):
            return {key: self.align(value) for key, value in datum.items()}
        elif isinstance(datum, list):
            return [self.align(value) for value in datum]
        elif isinstance(datum, tuple):
            return tuple(self.align(value) for value in datum)
        elif isinstance(datum, torch.Tensor):
            return datum.to(self.device)
        return datum

    def save(self, prefix: str) -> None:
        """
        Save the model.

        :param prefix: The prefix for the model file
        """
        model_path = os.path.join(self.output_dir, f"{prefix}.pt")
        optimizer_path = os.path.join(self.output_dir, f"{prefix}_optimizer.pt")
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)

    def load(self, prefix: str) -> None:
        """
        Load the model.

        :param prefix: The prefix for the model file
        """
        model_path = os.path.join(self.output_dir, f"{prefix}.pt")
        optimizer_path = os.path.join(self.output_dir, f"{prefix}_optimizer.pt")
        if hasattr(self.model, "best"):
            self.model.best()
        self.model.load_state_dict(torch.load(model_path))
        self.optimizer.load_state_dict(torch.load(optimizer_path))
        self.total_epochs = int(prefix)

    def resume(self) -> bool:
        """
        Resume training from the last epoch.

        :return: Whether the model was successfully resumed
        """
        model_files = os.listdir(self.output_dir)
        model_files = [model_file for model_file in model_files if model_file.endswith(".pt") and "_optimizer" not in model_file]
        model_epochs = [int(model_file.split(".")[0]) for model_file in model_files]
        for model_epoch in reversed(sorted(model_epochs)):
            optimizer_file = f"{model_epoch}_optimizer.pt"
            optimizer_path = os.path.join(self.output_dir, optimizer_file)
            if os.path.exists(optimizer_path):
                self.load(str(model_epoch))
                return True
        return False

    def __call__(self, num_epochs: int=10) -> None:
        """
        Train the model.

        :param num_epochs: The number of epochs to train for
        """
        if self.use_wandb:
            wandb.init(
                project=self.project_name,
                entity=self.wandb_entity,
                config={
                    "model": summarize_module(self.model),
                    "optimizer": summarize_optimizer(self.optimizer),
                }
            )

        num_steps_per_epoch = floor(len(self.training) / self.training_dataset_batch_size)
        scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            epochs=num_epochs,
            steps_per_epoch=num_steps_per_epoch,
        )
        if hasattr(self.model, "best"):
            self.model.best()
        self.model.train()

        for epoch in tqdm(range(num_epochs), unit="epoch", desc="Training"):
            for batch in tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                self.optimizer.zero_grad()
                loss = self.loss(self.align(batch))
                loss.backward() # type: ignore[no-untyped-call]
                self.optimizer.step()

                if self.use_wandb:
                    wandb.log({"loss": loss.item(), "learning_rate": scheduler.get_last_lr()[0]})

                scheduler.step()

            self.model.eval()
            self.evaluate(epoch + self.total_epochs)
            self.model.train()

        self.total_epochs += num_epochs
        self.save(f"{self.total_epochs}")
