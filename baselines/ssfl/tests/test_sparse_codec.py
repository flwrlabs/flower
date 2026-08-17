"""Tests for Phase-2 static-mask sparse packing."""

from __future__ import annotations

import torch

from ssfl.fixtures import generate_small_mask_fixture
from ssfl.mask import apply_mask_to_state_dict
from ssfl.sparse_codec import (
    compression_ratio,
    pack_state_dict,
    packed_numel,
    unpack_state_dict,
)


def test_pack_unpack_roundtrip():
    fixture = generate_small_mask_fixture(seed=550, n_clients=2, dense_ratio=0.5)
    masks = fixture["masks"]
    state = apply_mask_to_state_dict(fixture["init_state"], masks)
    packed = pack_state_dict(state, masks)
    restored = unpack_state_dict(packed, masks)
    assert set(restored) == set(state)
    for key in state:
        assert torch.allclose(restored[key], state[key])


def test_packed_payload_shrinks_with_density():
    fixture = generate_small_mask_fixture(seed=1, n_clients=2, dense_ratio=0.5)
    masks = fixture["masks"]
    state = apply_mask_to_state_dict(fixture["init_state"], masks)
    packed = pack_state_dict(state, masks)
    ratio = compression_ratio(state, packed)
    assert ratio < 0.85
    assert packed_numel(packed) < sum(t.numel() for t in state.values())

    # Masked layers must be 1-D active values only.
    for name, mask in masks.items():
        assert packed[name].ndim == 1
        assert packed[name].numel() == int(mask.sum().item())


def test_unpack_rejects_stale_packed_length():
    fixture = generate_small_mask_fixture(seed=2, n_clients=2, dense_ratio=0.5)
    masks = fixture["masks"]
    state = apply_mask_to_state_dict(fixture["init_state"], masks)
    packed = pack_state_dict(state, masks)
    # Corrupt one packed tensor.
    first_key = next(iter(masks))
    packed[first_key] = packed[first_key][: max(1, packed[first_key].numel() // 2)]
    try:
        unpack_state_dict(packed, masks)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Packed length mismatch" in str(exc)


def test_dense_sparse_local_update_equivalence():
    """Packed transport must not change numerical training updates."""
    from torch.utils.data import DataLoader, TensorDataset

    from ssfl.fixtures import generate_stage_oracle
    from ssfl.model import create_model, num_classes_for_dataset
    from ssfl.training import train_local

    oracle = generate_stage_oracle(seed=7, n_clients=2, dense_ratio=0.5)
    masks = oracle["masks"]
    masked_init = oracle["masked_init"]
    batch_x, batch_y = oracle["client_batches"][0]
    loader = DataLoader(
        TensorDataset(batch_x.repeat(2, 1, 1, 1), batch_y.repeat(2)),
        batch_size=16,
        shuffle=False,
    )

    def run_once():
        model = create_model("resnet18", num_classes_for_dataset("cifar10"))
        model.load_state_dict(masked_init)
        train_local(
            model,
            loader,
            epochs=1,
            lr=0.1,
            momentum=0.0,
            weight_decay=0.0005,
            max_grad_norm=10.0,
            round_idx=1,
            device=torch.device("cpu"),
            masks=masks,
        )
        state = apply_mask_to_state_dict(model.state_dict(), masks)
        packed = pack_state_dict(state, masks)
        restored = unpack_state_dict(packed, masks)
        return state, restored

    dense_state, restored = run_once()
    for key in dense_state:
        assert torch.allclose(dense_state[key], restored[key], atol=0.0)
