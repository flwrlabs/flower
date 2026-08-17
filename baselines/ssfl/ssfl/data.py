"""Dataset loading for simulation and deployment modes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

from ssfl.partitioner import BalancedDirichletPartitioner, partition_data_dirichlet

CIFAR10_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR10_STD = [0.24703233, 0.24348505, 0.26158768]
CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]

_fds_cache: dict[tuple[Any, ...], FederatedDataset] = {}
_local_index_cache: dict[tuple[Any, ...], dict[int, list[int]]] = {}


def _dataset_hub_name(dataset_name: str) -> str:
    if dataset_name == "cifar10":
        return "uoft-cs/cifar10"
    if dataset_name == "cifar100":
        return "uoft-cs/cifar100"
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def label_column_for_dataset(dataset_name: str) -> str:
    """HuggingFace column used for class labels."""
    if dataset_name == "cifar10":
        return "label"
    if dataset_name == "cifar100":
        # uoft-cs/cifar100 exposes fine_label / coarse_label, not label.
        return "fine_label"
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def labels_from_batch(batch: dict) -> Any:
    """Extract class labels from a Flower/HF dict batch."""
    for key in ("label", "fine_label", "labels"):
        if key in batch:
            return batch[key]
    raise KeyError(f"No label column in batch keys: {list(batch.keys())}")


def _transforms(dataset_name: str, train: bool):
    if dataset_name == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    elif dataset_name == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if train:
        return Compose(
            [
                RandomCrop(32, padding=4),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean, std),
            ]
        )
    return Compose([ToTensor(), Normalize(mean, std)])


def _apply_transforms(batch, transform):
    # Flower Datasets CIFAR uses the "img" column.
    key = "img" if "img" in batch else "image"
    batch[key] = [transform(img) for img in batch[key]]
    return batch


def _normalize_data_path(data_path: str | None) -> str | None:
    if data_path is None:
        return None
    text = str(data_path).strip()
    if not text:
        return None
    return text


def _torchvision_cifar(dataset_name: str, root: str, train: bool, transform):
    root_path = Path(root).expanduser().resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    if dataset_name == "cifar10":
        return CIFAR10(
            root=str(root_path), train=train, download=True, transform=transform
        )
    if dataset_name == "cifar100":
        return CIFAR100(
            root=str(root_path), train=train, download=True, transform=transform
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _local_partition_indices(
    *,
    dataset_name: str,
    data_path: str,
    num_partitions: int,
    partition_alpha: float,
    seed: int,
) -> dict[int, list[int]]:
    key = (dataset_name, data_path, num_partitions, partition_alpha, seed)
    cached = _local_index_cache.get(key)
    if cached is not None:
        return cached

    # Labels only; avoid applying train augmentations while partitioning.
    base = _torchvision_cifar(dataset_name, data_path, train=True, transform=None)
    y_train = np.asarray(base.targets, dtype=np.int64)
    mapping, _ = partition_data_dirichlet(
        y_train,
        n_clients=num_partitions,
        alpha=partition_alpha,
        seed=seed,
    )
    _local_index_cache[key] = mapping
    return mapping


def load_federated_dataset(
    dataset_name: str,
    num_partitions: int,
    partition_alpha: float,
    seed: int,
) -> FederatedDataset:
    """Load and cache a partitioned Flower dataset."""
    key = (dataset_name, num_partitions, partition_alpha, seed)
    cached = _fds_cache.get(key)
    if cached is not None:
        return cached

    partitioner = BalancedDirichletPartitioner(
        num_partitions=num_partitions,
        partition_by=label_column_for_dataset(dataset_name),
        alpha=partition_alpha,
        seed=seed,
    )
    fds = FederatedDataset(
        dataset=_dataset_hub_name(dataset_name),
        partitioners={"train": partitioner},
        seed=seed,
    )
    _fds_cache.clear()
    _fds_cache[key] = fds
    return fds


def load_partition_dataloaders(
    *,
    dataset_name: str,
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    partition_alpha: float,
    seed: int,
    val_fraction: float = 0.0,
    data_path: str | None = None,
    max_partition_samples: int = 0,
) -> tuple[DataLoader, DataLoader | None]:
    """
    Load a client's train (and optional local val) DataLoader.

    Simulation (default): Flower Datasets + BalancedDirichletPartitioner.
    Deployment: set ``data-path`` to a local directory; CIFAR is loaded via
    torchvision from that root and partitioned with the same algorithm.
    """
    resolved_path = _normalize_data_path(data_path)
    if resolved_path is not None:
        return _load_local_partition_dataloaders(
            dataset_name=dataset_name,
            partition_id=partition_id,
            num_partitions=num_partitions,
            batch_size=batch_size,
            partition_alpha=partition_alpha,
            seed=seed,
            val_fraction=val_fraction,
            data_path=resolved_path,
            max_partition_samples=max_partition_samples,
        )

    fds = load_federated_dataset(dataset_name, num_partitions, partition_alpha, seed)
    partition = fds.load_partition(partition_id)
    if 0 < max_partition_samples < len(partition):
        partition = partition.select(range(max_partition_samples))
    train_transform = _transforms(dataset_name, train=True)

    if val_fraction > 0:
        split = partition.train_test_split(test_size=val_fraction, seed=seed)
        train_ds = split["train"].with_transform(
            lambda batch: _apply_transforms(batch, train_transform)
        )
        test_transform = _transforms(dataset_name, train=False)
        val_ds = split["test"].with_transform(
            lambda batch: _apply_transforms(batch, test_transform)
        )
        trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        return trainloader, valloader

    train_ds = partition.with_transform(
        lambda batch: _apply_transforms(batch, train_transform)
    )
    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    return trainloader, None


def _load_local_partition_dataloaders(
    *,
    dataset_name: str,
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    partition_alpha: float,
    seed: int,
    val_fraction: float,
    data_path: str,
    max_partition_samples: int,
) -> tuple[DataLoader, DataLoader | None]:
    if partition_id < 0 or partition_id >= num_partitions:
        raise ValueError(
            f"partition_id/client-id {partition_id} out of range for "
            f"num_partitions={num_partitions}"
        )
    indices = _local_partition_indices(
        dataset_name=dataset_name,
        data_path=data_path,
        num_partitions=num_partitions,
        partition_alpha=partition_alpha,
        seed=seed,
    )[partition_id]
    if max_partition_samples > 0:
        indices = indices[:max_partition_samples]

    train_transform = _transforms(dataset_name, train=True)
    full_train = _torchvision_cifar(
        dataset_name, data_path, train=True, transform=train_transform
    )

    if val_fraction > 0:
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")
        rng = np.random.default_rng(seed)
        shuffled = list(indices)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_fraction)))
        val_idx = shuffled[:n_val]
        train_idx = shuffled[n_val:]
        if not train_idx:
            raise RuntimeError(
                f"Client {partition_id} has no training samples after val split"
            )
        test_transform = _transforms(dataset_name, train=False)
        full_val = _torchvision_cifar(
            dataset_name, data_path, train=True, transform=test_transform
        )
        trainloader = DataLoader(
            Subset(full_train, train_idx), batch_size=batch_size, shuffle=True
        )
        valloader = DataLoader(
            Subset(full_val, val_idx), batch_size=batch_size, shuffle=False
        )
        return trainloader, valloader

    trainloader = DataLoader(
        Subset(full_train, indices), batch_size=batch_size, shuffle=True
    )
    return trainloader, None


def load_centralized_testloader(
    dataset_name: str,
    batch_size: int = 128,
    data_path: str | None = None,
) -> DataLoader:
    """Load the centralized test split."""
    resolved_path = _normalize_data_path(data_path)
    test_transform = _transforms(dataset_name, train=False)
    if resolved_path is not None:
        test_ds = _torchvision_cifar(
            dataset_name, resolved_path, train=False, transform=test_transform
        )
        return DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    ds = load_dataset(_dataset_hub_name(dataset_name), split="test")
    ds = ds.with_transform(lambda batch: _apply_transforms(batch, test_transform))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def first_batches(
    trainloader: DataLoader, n: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Take the first n batches from a dataloader (for saliency)."""
    batches = []
    iterator = iter(trainloader)
    for _ in range(n):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        if isinstance(batch, dict):
            key = "img" if "img" in batch else "image"
            batches.append((batch[key], labels_from_batch(batch)))
        else:
            batches.append((batch[0], batch[1]))
    return batches
