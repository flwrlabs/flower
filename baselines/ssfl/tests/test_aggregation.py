"""Tests for sample-weighted FedAvg."""

from __future__ import annotations

import torch

from ssfl.aggregation import fedavg_weighted


def test_fedavg_weighted_matches_manual():
    a = {"w": torch.tensor([1.0, 3.0])}
    b = {"w": torch.tensor([3.0, 1.0])}
    out = fedavg_weighted([(1.0, a), (3.0, b)])
    expected = (1.0 * a["w"] + 3.0 * b["w"]) / 4.0
    assert torch.allclose(out["w"], expected)
