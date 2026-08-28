import random
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def split_dataset_dirichlet_fixed_size(dataset, num_clients, alpha, seed=42):
    """Split CIFAR-10 using Dirichlet sampling with equal samples per client."""
    random.seed(seed)
    np.random.seed(seed)

    data_by_class = defaultdict(list)

    for idx in range(len(dataset)):
        _, label = dataset[idx]
        data_by_class[label].append(idx)

    for cls in data_by_class:
        random.shuffle(data_by_class[cls])

    total_samples = len(dataset)
    samples_per_client = total_samples // num_clients

    clients_indices = [[] for _ in range(num_clients)]
    client_sample_counts = [0] * num_clients

    # Generate Dirichlet proportions for each class
    for cls in sorted(data_by_class.keys()):
        cls_indices = data_by_class[cls]
        cls_total = len(cls_indices)

        proportions = np.random.dirichlet([alpha] * num_clients)
        allocations = (proportions * cls_total).astype(int)

        # Make allocations sum exactly to the number of samples
        while allocations.sum() < cls_total:
            allocations[np.argmax(proportions)] += 1

        while allocations.sum() > cls_total:
            largest = np.argmax(allocations)
            if allocations[largest] > 0:
                allocations[largest] -= 1

        start = 0

        for client_id, allocation in enumerate(allocations):
            remaining = samples_per_client - client_sample_counts[client_id]

            if remaining <= 0:
                continue

            take = min(allocation, remaining)

            selected = cls_indices[start:start + take]
            clients_indices[client_id].extend(selected)

            client_sample_counts[client_id] += take
            start += take

    # Fill clients that are still under the target size
    assigned = set()

    for client_indices in clients_indices:
        assigned.update(client_indices)

    leftovers = [
        idx
        for cls_indices in data_by_class.values()
        for idx in cls_indices
        if idx not in assigned
    ]

    random.shuffle(leftovers)

    for client_id in range(num_clients):
        while (
            client_sample_counts[client_id] < samples_per_client
            and leftovers
        ):
            idx = leftovers.pop()
            clients_indices[client_id].append(idx)
            client_sample_counts[client_id] += 1

    return clients_indices


def print_data_distribution(dataset, clients_indices):
    """Print class distribution for every client."""
    for client_id, indices in enumerate(clients_indices):
        labels = [dataset.targets[idx] for idx in indices]
        counts = Counter(labels)

        print(
            f"Client {client_id}: "
            f"Samples={len(indices)}, "
            f"Class distribution={dict(sorted(counts.items()))}"
        )


def load_data(
    partition_id,
    num_partitions,
    batch_size=128,
    alpha=0.3,
    seed=42,
):
    """Load the CIFAR-10 partition assigned to one Flower client."""

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
        trainset,
        num_clients=num_partitions,
        alpha=alpha,
        seed=seed,
    )

    client_indices = clients_indices[partition_id]

    train_subset = torch.utils.data.Subset(
        trainset,
        client_indices,
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
