"""Unit tests for the balanced Dirichlet partitioner."""

from __future__ import annotations

import numpy as np

from ssfl.partitioner import partition_data_dirichlet


def test_balanced_counts():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, size=1000)
    mapping, counts = partition_data_dirichlet(y, n_clients=10, alpha=0.3, seed=550)
    sizes = [len(mapping[i]) for i in range(10)]
    assert min(sizes) == max(sizes) == 100
    assert counts.shape == (10, 10)
    assert counts.sum() == 1000


def test_seed_reproducibility():
    y = np.arange(200) % 10
    a, _ = partition_data_dirichlet(y, n_clients=4, alpha=0.3, seed=123)
    b, _ = partition_data_dirichlet(y, n_clients=4, alpha=0.3, seed=123)
    c, _ = partition_data_dirichlet(y, n_clients=4, alpha=0.3, seed=999)
    assert a == b
    assert a != c


def test_legacy_partition_parity_when_available():
    """Compare against the parent repo partitioner if importable."""
    try:
        import sys
        from pathlib import Path

        root = None
        for parent in Path(__file__).resolve().parents:
            if (parent / "data_preprocessing" / "partition_utils.py").exists():
                root = parent
                break
        if root is None:
            return
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from data_preprocessing.partition_utils import (
            partition_data_dirichlet as legacy_partition,
        )
    except Exception:
        return  # Parent package not available in a Hub-only checkout.

    y = np.arange(500) % 10
    seed = 550
    np.random.seed(seed)
    legacy_map, legacy_counts = legacy_partition(y, n_clients=5, alpha=0.3)

    port_map, port_counts = partition_data_dirichlet(
        y, n_clients=5, alpha=0.3, seed=seed
    )
    assert port_map == legacy_map
    assert np.array_equal(port_counts, legacy_counts)
