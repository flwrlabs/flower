"""Deployment data-path loading tests."""

from __future__ import annotations

import os
from collections.abc import Sized
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from torch.utils.data import Subset

from ssfl.data import load_centralized_testloader, load_partition_dataloaders
from ssfl.partitioner import partition_data_dirichlet


@pytest.fixture(scope="module")
def cifar_root(tmp_path_factory) -> Path:
    # Prefer a persistent local-disk cache; NFS temp dirs make torchvision
    # downloads extremely slow on shared cluster hosts.
    env_root = os.environ.get("SSFL_TEST_CIFAR_ROOT", "").strip()
    if env_root:
        root = Path(env_root)
    else:
        local = Path(
            f"/tmp/{os.environ.get('USER', 'user')}/ssfl-flower/torchvision-cifar"
        )
        try:
            local.mkdir(parents=True, exist_ok=True)
            probe = local / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            root = local
        except OSError:
            root = tmp_path_factory.mktemp("cifar-data")
    root.mkdir(parents=True, exist_ok=True)
    load_centralized_testloader("cifar10", batch_size=8, data_path=str(root))
    return root


def test_local_partitions_are_balanced_and_disjoint(cifar_root: Path) -> None:
    num_partitions = 4
    alpha = 0.3
    seed = 550
    loaders = [
        load_partition_dataloaders(
            dataset_name="cifar10",
            partition_id=i,
            num_partitions=num_partitions,
            batch_size=16,
            partition_alpha=alpha,
            seed=seed,
            data_path=str(cifar_root),
        )[0]
        for i in range(num_partitions)
    ]
    sizes = [len(cast(Sized, loader.dataset)) for loader in loaders]
    assert all(size == sizes[0] for size in sizes)
    assert sum(sizes) == 50000

    # Same seed/alpha as BalancedDirichletPartitioner path.
    from torchvision.datasets import CIFAR10

    labels = np.asarray(
        CIFAR10(root=str(cifar_root), train=True, download=False).targets
    )
    expected, _ = partition_data_dirichlet(
        labels, n_clients=num_partitions, alpha=alpha, seed=seed
    )
    for i, loader in enumerate(loaders):
        dataset = cast(Subset, loader.dataset)
        assert sorted(dataset.indices) == sorted(expected[i])


def test_local_testloader_has_official_size(cifar_root: Path) -> None:
    loader = load_centralized_testloader(
        "cifar10", batch_size=64, data_path=str(cifar_root)
    )
    assert len(cast(Sized, loader.dataset)) == 10000


def test_local_partition_sample_cap(cifar_root: Path) -> None:
    """Demo profiles can cap a client partition without changing paper runs."""
    loader, _ = load_partition_dataloaders(
        dataset_name="cifar10",
        partition_id=0,
        num_partitions=4,
        batch_size=16,
        partition_alpha=0.3,
        seed=550,
        data_path=str(cifar_root),
        max_partition_samples=512,
    )
    assert len(cast(Sized, loader.dataset)) == 512
