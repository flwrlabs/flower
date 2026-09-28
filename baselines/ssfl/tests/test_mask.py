"""Unit tests for mask creation and saliency aggregation."""

from __future__ import annotations

import pytest
import torch

from ssfl.mask import create_mask_from_scores, get_mean_saliency_scores, mask_digest


def test_uniform_mean_saliency():
    a = {"layer": torch.tensor([1.0, 3.0])}
    b = {"layer": torch.tensor([3.0, 1.0])}
    avg = get_mean_saliency_scores([a, b])
    assert torch.allclose(avg["layer"], torch.tensor([2.0, 2.0]))


def test_threshold_keeps_ties():
    # Three equal top scores: keep_ratio asks for 1, but all three tied survive.
    scores = {"conv": torch.tensor([5.0, 5.0, 5.0, 0.1])}
    masks, _ = create_mask_from_scores(scores, keep_ratio=0.25, device="cpu")
    mask = masks["conv.weight"]
    assert int(mask.sum().item()) == 3
    assert float(mask[-1].item()) == 0.0


def test_exact_topk_without_ties():
    scores = {"conv": torch.tensor([4.0, 3.0, 2.0, 1.0])}
    masks, density = create_mask_from_scores(scores, keep_ratio=0.5, device="cpu")
    mask = masks["conv.weight"]
    assert int(mask.sum().item()) == 2
    assert torch.equal(mask, torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert density["conv.weight"] == 0.5


def test_mask_digest_stable():
    scores = {"a": torch.tensor([1.0, 0.0, 2.0])}
    masks, _ = create_mask_from_scores(scores, keep_ratio=0.5, device="cpu")
    assert mask_digest(masks) == mask_digest(masks)


def test_keep_ratio_must_be_in_unit_interval():
    scores = {"conv": torch.tensor([4.0, 3.0, 2.0, 1.0])}
    with pytest.raises(ValueError, match="keep_ratio"):
        create_mask_from_scores(scores, keep_ratio=1.5, device="cpu")
    with pytest.raises(ValueError, match="keep_ratio"):
        create_mask_from_scores(scores, keep_ratio=-0.1, device="cpu")
