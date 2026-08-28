"""CIFAR-10 data loading and non-IID partitioning for FedSCS."""

import random
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def split_dataset_dirichlet_fixed_size(
    dataset: Any,
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> list[list[int]]:
    """Split CIFAR-10 into equal-size non-IID client partitions.

    A Dirichlet distribution controls the class proportions for each class,
    while client capacities are enforced so that every client receives the
    same number of training samples.
    """
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1.")
    if alpha <= 0:
        raise ValueError("alpha must be greater than 0.")
    if len(dataset) < num_clients:
        raise ValueError("num_clients cannot exceed the dataset size.")

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    data_by_class: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(dataset.targets):
        data_by_class[int(label)].append(idx)

    for indices in data_by_class.values():
        rng.shuffle(indices)

    total_samples = len(dataset)
    samples_per_client = total_samples // num_clients

    clients_indices: list[list[int]] = [[] for _ in range(num_clients)]
    remaining_capacity = [samples_per_client] * num_clients
    assigned_indices: set[int] = set()

    for cls in sorted(data_by_class):
        cls_indices = data_by_class[cls]
        cls_total = len(cls_indices)

        available_clients = [
            client_id
            for client_id in range(num_clients)
            if remaining_capacity[client_id] > 0
        ]

        if not available_clients:
            break

        proportions = np_rng.dirichlet(
            [alpha] * len(available_clients)
        )

        capacity = np.array(
            [remaining_capacity[i] for i in available_clients],
            dtype=np.int64,
        )

        allocations = np.zeros(
            len(available_clients),
            dtype=np.int64,
        )

        unassigned = cls_total

        while unassigned > 0 and capacity.sum() > 0:
            active = capacity > 0

            weights = proportions.copy()
            weights[~active] = 0.0

            if weights.sum() == 0:
                weights = active.astype(np.float64)

            weights /= weights.sum()

            draw = np_rng.multinomial(
                unassigned,
                weights,
            )

            draw = np.minimum(draw, capacity)
            assigned_now = int(draw.sum())

            if assigned_now == 0:
                break

            allocations += draw
            capacity -= draw
            unassigned -= assigned_now

        start = 0

        for position, client_id in enumerate(available_clients):
            count = int(allocations[position])

            if count <= 0:
                continue

            selected = cls_indices[start : start + count]

            clients_indices[client_id].extend(selected)
            assigned_indices.update(selected)

            remaining_capacity[client_id] -= count
            start += count

    # Assign any unassigned samples while respecting client capacity.
    remaining_indices = [
        idx
        for cls_indices in data_by_class.values()
        for idx in cls_indices
        if idx not in assigned_indices
    ]

    rng.shuffle(remaining_indices)

    position = 0

    for client_id in range(num_clients):
        needed = remaining_capacity[client_id]

        if needed <= 0:
            continue

        selected = remaining_indices[position : position + needed]

        clients_indices[client_id].extend(selected)
        assigned_indices.update(selected)

        remaining_capacity[client_id] -= len(selected)
        position += len(selected)

    if any(count != 0 for count in remaining_capacity):
        raise RuntimeError(
            "Unable to construct equal-size client partitions."
        )

    if len(assigned_indices) != samples_per_client * num_clients:
        raise RuntimeError(
            "Some training samples were not assigned to a client."
        )

    return clients_indices


def print_data_distribution(
    dataset: Any,
    clients_indices: list[list[int]],
) -> None:
    """Print the class distribution for every client."""
    targets = dataset.targets

    for client_id, indices in enumerate(clients_indices):
        labels = [int(targets[idx]) for idx in indices]
        counts = Counter(labels)

        print(
            f"Client {client_id}: "
            f"Samples={len(indices)}, "
            f"Class distribution={dict(sorted(counts.items()))}"
        )


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int = 128,
    alpha: float = 0.3,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Load the CIFAR-10 data assigned to one Flower client."""
    if num_partitions < 1:
        raise ValueError("num_partitions must be at least 1.")

    if not 0 <= partition_id < num_partitions:
        raise ValueError(
            f"partition_id must be in [0, {num_partitions - 1}], "
            f"got {partition_id}."
        )

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    transform = transforms.ToTensor()

    trainset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    testset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    clients_indices = split_dataset_dirichlet_fixed_size(
        dataset=trainset,
        num_clients=num_partitions,
        alpha=alpha,
        seed=seed,
    )

    train_subset = Subset(
        trainset,
        clients_indices[partition_id],
    )

    trainloader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
    )

    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
    )

    return trainloader, testloader
