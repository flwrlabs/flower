"""Dataset utilities matching the original GHBM CIFAR partitioning."""

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from numpy.random import RandomState
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import (
    Compose,
    ConvertImageDtype,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

from ghbm.config import DatasetName

DATA_ROOT = "./datasets"
PARTITION_SEED = 42
TRAIN_VAL_SPLIT_SEED = 42

TRAINSET_CACHE = {}
TESTSET_CACHE = {}
PARTITION_CACHE = {}
FDS_CACHE = {}
TEST_PARTITION_CACHE = {}

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
TorchvisionDataset: TypeAlias = type[CIFAR10] | type[CIFAR100]  # noqa: UP040


def _get_dataset_class(dataset_name: DatasetName) -> TorchvisionDataset:
    dataset_map = {
        DatasetName.CIFAR10: CIFAR10,
        DatasetName.CIFAR100: CIFAR100,
    }
    if dataset_name not in dataset_map:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return dataset_map[dataset_name]


def _get_partition_label_column(dataset_name: DatasetName) -> str:
    """Return the class-label column used by Flower Datasets."""
    label_column_map = {
        DatasetName.CIFAR10: "label",
        DatasetName.CIFAR100: "fine_label",
    }
    if dataset_name not in label_column_map:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return label_column_map[dataset_name]


def get_num_classes(dataset_name: DatasetName) -> int:
    """Return the number of classes for a supported dataset."""
    if dataset_name is DatasetName.CIFAR10:
        return 10
    if dataset_name is DatasetName.CIFAR100:
        return 100
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _get_train_transforms() -> Compose:
    return Compose(
        [
            ToTensor(),
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ConvertImageDtype(torch.float),
            Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


def _get_test_transforms() -> Compose:
    return Compose(
        [
            ToTensor(),
            ConvertImageDtype(torch.float),
            Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


class TransformedSubset(Dataset):
    """Apply a transform to a subset returned by torchvision datasets."""

    def __init__(self, subset: Subset, transform: Callable) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples in the subset."""
        return len(self.subset)

    def __getitem__(self, index):
        """Return one transformed sample as a Flower-style batch item."""
        image, label = self.subset[index]
        return {"img": self.transform(image), "label": label}


def _apply_hf_transforms(dataset, transform: Callable, label_column: str):
    """Apply torchvision transforms to a Hugging Face dataset split."""

    def transform_batch(batch):
        batch["img"] = [transform(img) for img in batch["img"]]
        batch["label"] = batch[label_column]
        return batch

    return dataset.with_transform(transform_batch)


def _load_trainset(dataset_name: DatasetName):
    if dataset_name not in TRAINSET_CACHE:
        dataset_class = _get_dataset_class(dataset_name)
        TRAINSET_CACHE[dataset_name] = dataset_class(
            root=DATA_ROOT,
            train=True,
            download=True,
        )
    return TRAINSET_CACHE[dataset_name]


def _load_testset(dataset_name: DatasetName):
    if dataset_name not in TESTSET_CACHE:
        dataset_class = _get_dataset_class(dataset_name)
        TESTSET_CACHE[dataset_name] = dataset_class(
            root=DATA_ROOT,
            train=False,
            download=True,
        )
    return TESTSET_CACHE[dataset_name]


def _create_from_contiguous_shards(
    targets: np.ndarray, num_clients: int, shard_size: int
) -> list[np.ndarray]:
    shard_start_index = list(range(0, len(targets), shard_size))
    random_state = RandomState(PARTITION_SEED)
    random_state.shuffle(shard_start_index)

    partitions = []
    for client_shard_starts in np.array_split(shard_start_index, num_clients):
        client_indices = [
            np.arange(
                start,
                min(
                    start + shard_size,
                    len(targets),
                ),
            )
            for start in client_shard_starts.tolist()
        ]
        if client_indices:
            partitions.append(np.concatenate(client_indices, axis=0))
        else:
            partitions.append(np.array([], dtype=np.int64))
    return partitions


def _create_non_iid_partitions(
    targets: np.ndarray, num_clients: int
) -> list[np.ndarray]:
    sorted_indices = np.argsort(targets)
    shard_size = max(1, len(targets) // num_clients)
    sorted_targets = targets[sorted_indices]
    sorted_partition_indices = _create_from_contiguous_shards(
        sorted_targets, num_clients, shard_size
    )
    return [sorted_indices[partition] for partition in sorted_partition_indices]


def _get_partition_indices(
    dataset_name: DatasetName, num_partitions: int, dirichlet_alpha: float
) -> list[np.ndarray]:
    cache_key = (dataset_name, num_partitions, float(dirichlet_alpha))
    if cache_key not in PARTITION_CACHE:
        if dirichlet_alpha != 0:
            raise ValueError(
                "_get_partition_indices is only used for the alpha=0 shard split"
            )
        trainset = _load_trainset(dataset_name)
        targets = np.array(trainset.targets)
        PARTITION_CACHE[cache_key] = _create_non_iid_partitions(targets, num_partitions)
    return PARTITION_CACHE[cache_key]


def _get_test_partition_indices(
    dataset_name: DatasetName, num_partitions: int
) -> list[np.ndarray]:
    """Create deterministic disjoint partitions of the global test split."""
    cache_key = (dataset_name, num_partitions)
    if cache_key not in TEST_PARTITION_CACHE:
        testset = _load_testset(dataset_name)
        random_state = RandomState(PARTITION_SEED)
        shuffled_indices = random_state.permutation(len(testset))
        TEST_PARTITION_CACHE[cache_key] = [
            np.array(split, dtype=np.int64)
            for split in np.array_split(shuffled_indices, num_partitions)
        ]
    return TEST_PARTITION_CACHE[cache_key]


def _get_federated_dataset(
    dataset_name: DatasetName, num_partitions: int, dirichlet_alpha: float
) -> FederatedDataset:
    cache_key = (dataset_name, num_partitions, float(dirichlet_alpha))
    if cache_key not in FDS_CACHE:
        dataset_id = f"uoft-cs/{dataset_name}"
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by=_get_partition_label_column(dataset_name),
            alpha=dirichlet_alpha,
            seed=PARTITION_SEED,
        )
        FDS_CACHE[cache_key] = FederatedDataset(
            dataset=dataset_id,
            partitioners={"train": partitioner},
        )
    return FDS_CACHE[cache_key]


def load_data(
    partition_id: int,
    num_partitions: int,
    dataset_name: DatasetName,
    dirichlet_alpha: float,
    batch_size: int = 32,
):
    """Load a client partition using the original GHBM CIFAR splits."""
    train_transform = _get_train_transforms()
    test_transform = _get_test_transforms()
    assert dirichlet_alpha >= 0, "dirichlet_alpha must be non-negative"
    if dirichlet_alpha > 0:
        fds = _get_federated_dataset(dataset_name, num_partitions, dirichlet_alpha)
        label_column = _get_partition_label_column(dataset_name)
        testset = _load_testset(dataset_name)
        test_partition_indices = _get_test_partition_indices(
            dataset_name, num_partitions
        )[partition_id]
        partition = fds.load_partition(partition_id)
        partition_train_test = partition.train_test_split(
            test_size=0.2, seed=TRAIN_VAL_SPLIT_SEED + partition_id
        )
        train_dataset = _apply_hf_transforms(
            partition_train_test["train"], train_transform, label_column
        )
        val_dataset = _apply_hf_transforms(
            partition_train_test["test"], test_transform, label_column
        )
        test_dataset = TransformedSubset(
            Subset(testset, test_partition_indices.tolist()), test_transform
        )

        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(val_dataset, batch_size=batch_size)
        testloader = DataLoader(test_dataset, batch_size=batch_size)
        return trainloader, valloader, testloader

    trainset = _load_trainset(dataset_name)
    testset = _load_testset(dataset_name)

    partition_indices = _get_partition_indices(
        dataset_name, num_partitions, dirichlet_alpha
    )[partition_id]
    split_rng = RandomState(TRAIN_VAL_SPLIT_SEED + partition_id)
    shuffled_indices = split_rng.permutation(partition_indices)
    split_point = int(0.8 * len(shuffled_indices))
    train_indices = shuffled_indices[:split_point]
    val_indices = shuffled_indices[split_point:]
    test_partition_indices = _get_test_partition_indices(dataset_name, num_partitions)[
        partition_id
    ]

    train_subset = TransformedSubset(
        Subset(trainset, train_indices.tolist()), train_transform
    )
    val_subset = TransformedSubset(
        Subset(trainset, val_indices.tolist()), test_transform
    )
    test_subset = TransformedSubset(
        Subset(testset, test_partition_indices.tolist()), test_transform
    )

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_subset, batch_size=batch_size)
    testloader = DataLoader(test_subset, batch_size=batch_size)
    return trainloader, valloader, testloader
