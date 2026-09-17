"""Data generation and loading for the FedHT baseline.

Simulation I:  100 clients, 100 samples each, d=1000, sparse linear regression,
               alpha=0.1, beta=0.1 (Section V of the paper).
Simulation II: same structure, binary labels, sparse logistic regression,
               alpha=1.0, beta=1.0.
MNIST is loaded via flwr-datasets with a pathological (2-class-per-client) split.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def generate_simulation_I(
    num_clients: int = 100,
    num_samples: int = 100,
    d: int = 1000,
    true_sparsity: int = 100,
    alpha: float = 0.1,
    beta: float = 0.1,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate non-IID data for Simulation I (sparse linear regression).

    Each client gets its own local coefficient vector where the first
    true_sparsity elements are drawn from N(u_i, 1) and the rest are zero.
    """
    rng = np.random.default_rng(seed)
    cov_diag = np.array([1.0 / (i + 1) ** 1.2 for i in range(d)])

    datasets = []
    for _ in range(num_clients):
        u_i = rng.normal(0.1, alpha)
        B_i = rng.normal(0.0, beta)

        x_i = np.zeros(d)
        x_i[:true_sparsity] = rng.normal(u_i, 1.0, size=true_sparsity)

        v_i = rng.normal(B_i, 1.0, size=d)
        Z_i = rng.normal(v_i, np.sqrt(cov_diag), size=(num_samples, d))

        b_i = rng.normal(u_i, 1.0, size=num_samples)
        y_i = Z_i @ x_i + b_i

        datasets.append((Z_i.astype(np.float32), y_i.astype(np.float32)))

    return datasets


def generate_simulation_II(
    num_clients: int = 100,
    num_samples: int = 1000,
    d: int = 1000,
    true_sparsity: int = 100,
    alpha: float = 1.0,
    beta: float = 1.0,
    top_k_positive: int = 100,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate non-IID data for Simulation II (sparse logistic regression).

    Same covariate structure as Simulation I. Labels are binary: the top
    top_k_positive samples by sigmoid score are assigned label 1.
    """
    rng = np.random.default_rng(seed)
    cov_diag = np.array([1.0 / (i + 1) ** 1.2 for i in range(d)])

    datasets = []
    for _ in range(num_clients):
        u_i = rng.normal(0.1, alpha)
        B_i = rng.normal(0.0, beta)

        x_i = np.zeros(d)
        x_i[:true_sparsity] = rng.normal(u_i, 1.0, size=true_sparsity)

        v_i = rng.normal(B_i, 1.0, size=d)
        Z_i = rng.normal(v_i, np.sqrt(cov_diag), size=(num_samples, d))

        b_i = rng.normal(u_i, 1.0, size=num_samples)
        scores = Z_i @ x_i + b_i
        sigmoid_scores = 1.0 / (1.0 + np.exp(-scores))

        y_i = np.zeros(num_samples, dtype=np.float32)
        top_indices = np.argsort(sigmoid_scores)[-top_k_positive:]
        y_i[top_indices] = 1.0

        datasets.append((Z_i.astype(np.float32), y_i))

    return datasets


def make_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 20,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Split (X, y) into train and validation DataLoaders."""
    rng = np.random.default_rng(seed)
    n = len(X)
    indices = rng.permutation(n)
    split = max(1, int(n * val_ratio))
    val_idx, train_idx = indices[:split], indices[split:]

    X_t = torch.tensor(X[train_idx])
    y_t = torch.tensor(y[train_idx])
    X_v = torch.tensor(X[val_idx])
    y_v = torch.tensor(y[val_idx])

    train_loader = DataLoader(
        TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(X_v, y_v), batch_size=batch_size)
    return train_loader, val_loader


def load_mnist_partition(
    partition_id: int,
    num_partitions: int,
    batch_size: int = 64,
    val_ratio: float = 0.1,
) -> tuple[DataLoader, DataLoader]:
    """Load one MNIST partition using flwr-datasets.

    Each client receives samples from exactly 2 of the 10 digit classes
    (PathologicalPartitioner), giving roughly 600 samples per client.
    """
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import PathologicalPartitioner

    fds = FederatedDataset(
        dataset="mnist",
        partitioners={
            "train": PathologicalPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                num_classes_per_partition=2,
            )
        },
    )
    partition = fds.load_partition(partition_id)
    partition = partition.with_format("numpy")

    X = (
        partition["image"].reshape(len(partition["image"]), -1).astype(np.float32)
        / 255.0
    )
    y = partition["label"].astype(np.int64)

    return make_loaders(X, y, batch_size=batch_size, val_ratio=val_ratio)
