"""Unit tests for corruptions, corrupted-client selection and partitioning."""

import numpy as np
import pytest
import torch
from datasets import Dataset

from fedagain.corruptions import CORRUPTIONS, apply_corruption, parse_corruption_types
from fedagain.dataset import LabelSkewPartitioner, corrupt_images, corrupted_clients


@pytest.mark.parametrize("name", list(CORRUPTIONS))
@pytest.mark.parametrize("channels", [1, 3])
def test_corruptions_keep_shape_and_range(name, channels):
    gen = torch.Generator().manual_seed(0)
    x = torch.rand(channels, 32, 32, generator=gen)
    out = apply_corruption(x, name, severity=3, gen=gen)
    assert out.shape == x.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert not torch.equal(out, x)


def test_parse_corruption_types():
    assert parse_corruption_types("all") == list(CORRUPTIONS)
    assert parse_corruption_types("contrast, shot_noise") == ["contrast", "shot_noise"]
    with pytest.raises(ValueError):
        parse_corruption_types("rain")


def test_corrupted_clients_deterministic():
    first = corrupted_clients(10, 0.3, seed=42)
    assert first == corrupted_clients(10, 0.3, seed=42)
    assert len(first) == 3 and len(set(first)) == 3
    assert corrupted_clients(10, 0.0, seed=42) == []


def test_label_skew_partitioner():
    labels = np.repeat(np.arange(10), 1000)
    partitioner = LabelSkewPartitioner(
        num_partitions=10, partition_by="label", num_primary=2, primary_share=0.8
    )
    partitioner.dataset = Dataset.from_dict({"label": labels.tolist()})
    sizes, seen = [], []
    for k in range(10):
        part = np.asarray(partitioner.load_partition(k)["label"])
        sizes.append(len(part))
        seen.extend(part.tolist())
        primary = partitioner.primary_classes(k, 10)
        # 80% of each primary class is split across its 2 primary clients
        assert np.isin(part, primary).mean() == pytest.approx(0.8, abs=0.02)
    assert sum(sizes) == len(labels)
    assert sorted(seen) == sorted(labels.tolist())


def test_corrupt_images_ratio():
    gen = torch.Generator().manual_seed(0)
    images = torch.rand(40, 3, 8, 8, generator=gen)
    original = images.clone()
    config = {
        "corruption-types": "all",
        "corruption-severity": 3,
        "corruption-ratio": 0.5,
    }
    out = corrupt_images(images, config, torch.Generator().manual_seed(1))
    changed = (out != original).flatten(1).any(dim=1)
    assert int(changed.sum()) == 20
