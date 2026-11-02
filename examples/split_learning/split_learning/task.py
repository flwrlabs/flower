"""Models and data utilities for the split learning demo."""

from functools import lru_cache
from typing import Iterable, Tuple

import torch
import torch.nn as nn
from flwr.common import Parameters, ndarrays_to_parameters
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class ClientNet(nn.Module):
    """Lower part of the split model that stays on each client."""

    def __init__(self, embedding_size: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, embedding_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ServerNet(nn.Module):
    """Upper part of the split model that stays on the server."""

    def __init__(self, embedding_size: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def model_to_parameters(model: nn.Module) -> Parameters:
    """Convert a PyTorch state dict into Flower `Parameters`."""
    ndarrays = [param.detach().cpu().numpy() for _, param in model.state_dict().items()]
    return ndarrays_to_parameters(ndarrays)


@lru_cache(maxsize=1)
def _federated_dataset(num_partitions: int) -> FederatedDataset:
    partitioner = IidPartitioner(num_partitions=num_partitions)
    return FederatedDataset(
        dataset="ylecun/mnist",
        partitioners={"train": partitioner},
    )


def _apply_transforms(batch):
    pytorch_transforms = Compose(
        [
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
        ]
    )
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    return batch


def load_data(
    partition_id: int, num_partitions: int, batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """Load and partition MNIST for a given client."""
    fds = _federated_dataset(num_partitions=num_partitions)
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    partition_train_test = partition_train_test.with_transform(_apply_transforms)

    trainloader = DataLoader(
        partition_train_test["train"],
        batch_size=batch_size,
        shuffle=True,
    )
    testloader = DataLoader(
        partition_train_test["test"],
        batch_size=batch_size,
    )
    return trainloader, testloader


def count_parameters(modules: Iterable[nn.Module]) -> int:
    """Count trainable parameters across a collection of modules."""
    return sum(
        p.numel()
        for module in modules
        for p in module.parameters()
        if p.requires_grad
    )
