"""Balanced Dirichlet partitioner matching the legacy SSFL algorithm."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from datasets import Dataset
from flwr_datasets.partitioner import Partitioner


class BalancedDirichletPartitioner(Partitioner):
    """
    Dirichlet non-IID partitioner with equal sample counts per client.

    This mirrors `partition_data_dirichlet` in the legacy SSFL repository:
    1. Fix each client's sample budget.
    2. Sample per-client class priors from Dirichlet(alpha).
    3. Assign samples by drawing from available class pools.
    """

    def __init__(
        self,
        num_partitions: int,
        partition_by: str = "label",
        alpha: float = 0.3,
        seed: int = 550,
    ) -> None:
        super().__init__()
        if num_partitions <= 0:
            raise ValueError("num_partitions must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self._num_partitions = num_partitions
        self._partition_by = partition_by
        self._alpha = float(alpha)
        self._seed = seed
        self._partition_id_to_indices: dict[int, list[int]] = {}
        self._determined = False

    def load_partition(self, partition_id: int) -> Dataset:
        """Load one client partition by identifier."""
        self._determine_indices_if_needed()
        # The inherited dataset property is a HuggingFace Dataset at runtime.
        # pylint: disable-next=no-member
        return self.dataset.select(self._partition_id_to_indices[partition_id])

    @property
    def num_partitions(self) -> int:
        """Return the configured number of partitions."""
        self._determine_indices_if_needed()
        return self._num_partitions

    def partition_indices(self) -> dict[int, list[int]]:
        """Return a copy of the client-to-sample index mapping."""
        self._determine_indices_if_needed()
        return {k: list(v) for k, v in self._partition_id_to_indices.items()}

    def _determine_indices_if_needed(self) -> None:
        if self._determined:
            return
        y_train = np.array(self.dataset[self._partition_by])
        mapping, _ = partition_data_dirichlet(
            y_train,
            n_clients=self._num_partitions,
            alpha=self._alpha,
            seed=self._seed,
        )
        self._partition_id_to_indices = mapping
        self._determined = True


def partition_data_dirichlet(
    y_train: np.ndarray,
    n_clients: int,
    alpha: float,
    seed: int | None = None,
) -> tuple[dict[int, list[int]], np.ndarray]:
    """
    Legacy SSFL Dirichlet partition with balanced sample counts.

    Returns
    -------
    tuple[dict[int, list[int]], np.ndarray]
        Client-to-sample index mapping and per-client class counts.
    """
    # Use global np.random to match the legacy SSFL partitioner exactly.
    if seed is not None:
        np.random.seed(seed)

    n_classes = len(np.unique(y_train))
    n_samples = len(y_train)

    samples_per_client = n_samples // n_clients
    client_sample_counts = cast(Any, np.full(n_clients, samples_per_client))
    client_sample_counts[: n_samples % n_clients] += 1

    client_class_priors = cast(
        Any,
        np.random.dirichlet(alpha=np.repeat(alpha, n_classes), size=n_clients),
    )
    class_pools = cast(
        list[list[int]],
        [list(np.where(y_train == i)[0]) for i in range(n_classes)],
    )
    for pool in class_pools:
        np.random.shuffle(pool)

    client_indices_map: dict[int, list[int]] = {i: [] for i in range(n_clients)}
    client_slots = cast(Any, np.repeat(np.arange(n_clients), client_sample_counts))
    np.random.shuffle(client_slots)

    for client_idx in client_slots.tolist():
        priors = client_class_priors[client_idx]
        available_classes = [k for k, pool in enumerate(class_pools) if len(pool) > 0]
        if not available_classes:
            continue
        probs = priors[available_classes]
        normalized = probs / np.sum(probs)
        chosen_class = int(np.random.choice(available_classes, p=normalized))
        sample_idx = class_pools[chosen_class].pop()
        client_indices_map[int(client_idx)].append(int(sample_idx))

    final_class_counts = cast(Any, np.zeros((n_clients, n_classes), dtype=int))
    for client_id, indices in client_indices_map.items():
        if indices:
            labels = y_train[np.array(indices, dtype=int)]
            final_class_counts[client_id, :] = np.bincount(labels, minlength=n_classes)

    return client_indices_map, final_class_counts
