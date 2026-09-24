"""FedSaSync: Semi-asynchronous Federated Learning in Flower."""

import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

FDS = None  # Cache FederatedDataset


def load_data(
    partition_id: int, num_partitions: int, dataset_name: str = "uoft-cs/cifar10"
):
    """Load a dataset partition (CIFAR10 or MNIST) and return train/test DataLoaders."""
    # Only initialize `FederatedDataset` once
    global FDS  # pylint: disable=global-statement
    if FDS is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        FDS = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )
    partition = FDS.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    if dataset_name == "uoft-cs/cifar10":
        pytorch_transforms = Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
    elif dataset_name == "ylecun/mnist":
        pytorch_transforms = Compose(
            [ToTensor(), Normalize((0.1307,), (0.3081,))]
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    def apply_transforms(batch):
        image = "img" if dataset_name == "uoft-cs/cifar10" else "image"
        batch[image] = [pytorch_transforms(img) for img in batch[image]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"],
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    testloader = DataLoader(
        partition_train_test["test"],
        batch_size=32,
        generator=torch.Generator().manual_seed(42),
    )
    return trainloader, testloader
