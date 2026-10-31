"""Deterministic synthetic dataset utilities for verifiable aggregation baseline."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def _make_partition(
    partition_id: int,
    num_partitions: int,
    *,
    num_train_examples: int,
    num_val_examples: int,
    num_features: int,
    base_seed: int,
) -> tuple[TensorDataset, TensorDataset]:
    """Create deterministic train/val tensors for one partition."""
    total_examples = num_train_examples + num_val_examples

    partition_seed = base_seed + (partition_id * 9973)
    rng = np.random.default_rng(partition_seed)

    part_position = partition_id / max(num_partitions - 1, 1)
    shift = np.linspace(-0.3, 0.3, num_features, dtype=np.float32) * (
        2.0 * part_position - 1.0
    )

    x = rng.normal(loc=0.0, scale=1.0, size=(total_examples, num_features)).astype(
        np.float32
    )
    x = x + shift

    w = np.linspace(-1.0, 1.0, num_features, dtype=np.float32)
    logits = x @ w + 0.15 * (partition_id % 3)
    y = (logits > 0.0).astype(np.int64)

    x_train = torch.from_numpy(x[:num_train_examples])
    y_train = torch.from_numpy(y[:num_train_examples])
    x_val = torch.from_numpy(x[num_train_examples:])
    y_val = torch.from_numpy(y[num_train_examples:])

    trainset = TensorDataset(x_train, y_train)
    valset = TensorDataset(x_val, y_val)
    return trainset, valset


def load_data(
    partition_id: int,
    num_partitions: int,
    *,
    num_train_examples: int,
    num_val_examples: int,
    num_features: int,
    batch_size: int,
    base_seed: int,
) -> tuple[DataLoader, DataLoader]:
    """Load deterministic local train and val data loaders for one client."""
    trainset, valset = _make_partition(
        partition_id=partition_id,
        num_partitions=num_partitions,
        num_train_examples=num_train_examples,
        num_val_examples=num_val_examples,
        num_features=num_features,
        base_seed=base_seed,
    )

    generator = torch.Generator().manual_seed(base_seed + partition_id)
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    return trainloader, valloader
