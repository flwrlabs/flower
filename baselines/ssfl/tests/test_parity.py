"""Parity tests against legacy SSFL helpers and regenerated stage oracles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ssfl.fixtures import generate_stage_oracle
from ssfl.mask import create_mask_from_scores, get_mean_saliency_scores
from ssfl.partitioner import partition_data_dirichlet


def _legacy_root() -> Path | None:
    """Find the original SSFL repo if this baseline sits next to it."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "api" / "sparsity" / "saliency_utils.py").exists():
            return parent
    return None


def _require_legacy_root() -> Path:
    root = _legacy_root()
    if root is None:
        pytest.skip("Legacy SSFL package not importable")
    return root


def test_stage_oracle_is_deterministic():
    a = generate_stage_oracle(seed=550, n_clients=2, dense_ratio=0.5)
    b = generate_stage_oracle(seed=550, n_clients=2, dense_ratio=0.5)
    assert a["mask_digest"] == b["mask_digest"]
    assert a["active_params"] == b["active_params"]
    for key in a["masks"]:
        assert torch.equal(a["masks"][key], b["masks"][key])
    for key in a["aggregated"]:
        assert torch.allclose(a["aggregated"][key], b["aggregated"][key], atol=1e-6)


def test_legacy_mask_utils_parity_when_available():
    try:
        import sys

        root = _require_legacy_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from api.sparsity.saliency_utils import (
            create_mask_from_scores as legacy_create_mask,
        )
        from api.sparsity.saliency_utils import (
            get_mean_saliency_scores as legacy_mean,
        )
    except Exception:
        pytest.skip("Legacy SSFL package not importable")

    scores = [
        {"layer": torch.tensor([1.0, 4.0, 0.5, 2.0])},
        {"layer": torch.tensor([3.0, 0.0, 1.5, 2.0])},
    ]
    port_avg = get_mean_saliency_scores(scores)
    legacy_avg = legacy_mean(scores)
    assert torch.allclose(port_avg["layer"], legacy_avg["layer"])

    port_masks, _ = create_mask_from_scores(port_avg, keep_ratio=0.5, device="cpu")
    legacy_masks, _ = legacy_create_mask(legacy_avg, keep_ratio=0.5, device="cpu")
    assert set(port_masks) == set(legacy_masks)
    for key in port_masks:
        assert torch.equal(port_masks[key], legacy_masks[key])


def test_legacy_partition_and_saliency_parity_when_available():
    try:
        import sys

        root = _require_legacy_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from data_preprocessing.partition_utils import (
            partition_data_dirichlet as legacy_partition,
        )
    except Exception:
        pytest.skip("Legacy partitioner not importable")

    y = np.arange(200) % 10
    seed = 550
    np.random.seed(seed)
    legacy_map, legacy_counts = legacy_partition(y, n_clients=4, alpha=0.3)
    port_map, port_counts = partition_data_dirichlet(
        y, n_clients=4, alpha=0.3, seed=seed
    )
    assert port_map == legacy_map
    assert np.array_equal(port_counts, legacy_counts)

    # Tiny model saliency formula check against port helper already covered;
    # ensure scores are finite/non-negative on a real ResNet batch from oracle.
    oracle = generate_stage_oracle(seed=123, n_clients=2, dense_ratio=0.5)
    for scores in oracle["local_scores"]:
        assert scores
        for tensor in scores.values():
            assert torch.all(tensor >= 0)
            assert torch.isfinite(tensor).all()


def test_oracle_mask_keep_ratio_approximately():
    oracle = generate_stage_oracle(seed=550, n_clients=2, dense_ratio=0.5)
    total = sum(int(m.numel()) for m in oracle["masks"].values())
    active = oracle["active_params"]
    # Ties can push density slightly above keep_ratio.
    assert active / total >= 0.5
    assert active / total < 0.6
