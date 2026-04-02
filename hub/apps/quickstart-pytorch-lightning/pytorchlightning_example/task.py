"""pytorchlightning_example: A Flower / PyTorch Lightning app."""

import logging
from typing import Any

import pytorch_lightning as pl
import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch import nn
from torch.nn import functional as F
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


class LitAutoEncoder(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28),
        )

    def forward(self, x) -> Any:
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self) -> Adam:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        x = train_batch["image"]
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        self._evaluate(batch, "val")

    def test_step(self, batch, batch_idx) -> None:
        self._evaluate(batch, "test")

    def _evaluate(self, batch, stage=None) -> None:
        x = batch["image"]
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [transforms.functional.to_tensor(img) for img in batch["image"]]
    return batch


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_partitions):
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id, "train")

    partition = partition.with_transform(apply_transforms)
    # 20 % for on federated evaluation
    partition_full = partition.train_test_split(test_size=0.2, seed=42)
    # 60 % for the federated train and 20 % for the federated validation (both in fit)
    partition_train_valid = partition_full["train"].train_test_split(
        train_size=0.75, seed=42
    )
    trainloader = DataLoader(
        partition_train_valid["train"],
        batch_size=32,
        shuffle=True,
        num_workers=2,
    )
    valloader = DataLoader(
        partition_train_valid["test"],
        batch_size=32,
        num_workers=2,
    )
    testloader = DataLoader(partition_full["test"], batch_size=32, num_workers=1)
    return trainloader, valloader, testloader
