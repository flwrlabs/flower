"""CIFAR-100 HuggingFace column compatibility."""

from __future__ import annotations

from ssfl.data import (
    label_column_for_dataset,
    labels_from_batch,
    load_federated_dataset,
)


def test_label_column_names() -> None:
    assert label_column_for_dataset("cifar10") == "label"
    assert label_column_for_dataset("cifar100") == "fine_label"


def test_labels_from_batch_prefers_available_keys() -> None:
    assert labels_from_batch({"fine_label": 3, "img": None}) == 3
    assert labels_from_batch({"label": 1, "img": None}) == 1


def test_cifar100_federated_dataset_partitions() -> None:
    fds = load_federated_dataset(
        dataset_name="cifar100",
        num_partitions=4,
        partition_alpha=0.3,
        seed=550,
    )
    part = fds.load_partition(0)
    assert "fine_label" in part.column_names
    assert "label" not in part.column_names
    assert len(part) == 50000 // 4
