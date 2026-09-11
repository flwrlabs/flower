"""Unit tests for local training / gradient clipping."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ssfl.model import SparseModel
from ssfl.training import train_local


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 3)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


def test_train_local_runs_and_clips():
    torch.manual_seed(0)
    model = SparseModel(TinyNet())
    x = torch.randn(16, 8)
    y = torch.randint(0, 3, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)

    loss, lr = train_local(
        model,
        loader,
        epochs=1,
        lr=0.1,
        momentum=0.0,
        weight_decay=0.0,
        max_grad_norm=10.0,
        round_idx=1,
        lr_scheduler_name="default",
        lr_decay=0.998,
        device=torch.device("cpu"),
    )
    assert loss >= 0.0
    assert abs(lr - 0.1 * (0.998**1)) < 1e-9


def test_load_state_dict_honors_strict():
    model = SparseModel(TinyNet())
    state = model.state_dict()
    result = model.load_state_dict(state, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []

    extra = dict(state)
    extra["not_a_real_key"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        model.load_state_dict(extra, strict=True)
    result = model.load_state_dict(extra, strict=False)
    assert "not_a_real_key" in result.unexpected_keys

    incomplete = {key: value for key, value in state.items() if key != "fc.weight"}
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(incomplete, strict=True)


def test_load_state_dict_strict_with_applied_masks():
    model = SparseModel(TinyNet())
    public_state = model.state_dict()
    model.apply_masks({"fc.weight": torch.ones_like(public_state["fc.weight"])})
    state = model.state_dict()
    assert all(not key.endswith(".weight_mask") for key in state)
    result = model.load_state_dict(state, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
