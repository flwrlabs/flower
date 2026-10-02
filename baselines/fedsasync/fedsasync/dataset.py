"""FedSaSync: Semi-asynchronous Federated Learning in Flower."""

import torch
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner

FDS = None  # Cache FederatedDataset


def get_partitioner(partitioner_type: str, num_partitions: int):
    """Get the partitioner based on the partitioner type with default/hardcoded values."""
    key = partitioner_type.lower()
    
    if key == "iid":
        return IidPartitioner(num_partitions=num_partitions)
    elif key == "dirichlet":
        return DirichletPartitioner(
            num_partitions=num_partitions,
            alpha=0.5,
            partition_by="label"
        )
    else:
        raise ValueError(f"Unknown partitioner type: {partitioner_type}")


def load_data(
        partition_id: int,
        num_partitions: int,
        dataset_name: str = "uoft-cs/cifar10",
        data_distribution: str = "iid",
        run_id: int = 1,
    ):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global FDS  # pylint: disable=global-statement
    if FDS is None:
        partitioner = get_partitioner(
            partitioner_type=data_distribution,
            num_partitions=num_partitions,
        )
        FDS = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )
    partition = FDS.load_partition(partition_id)
    seed = 42 + run_id
    
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=seed)

    if dataset_name == "uoft-cs/cifar10":
        pytorch_transforms = Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
    elif dataset_name == "ylecun/mnist":
        pytorch_transforms = Compose(
            [ToTensor(), Normalize((0.1307,), (0.3081,))]
        )
    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        image = "img" if dataset_name == "uoft-cs/cifar10" else "image"
        batch[image] = [pytorch_transforms(img) for img in batch[image]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"],
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed)
    )
    testloader = DataLoader(
        partition_train_test["test"],
        batch_size=32,
        generator=torch.Generator().manual_seed(seed)
    )
    return trainloader, testloader
