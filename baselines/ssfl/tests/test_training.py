"""Unit tests for local training / gradient clipping."""

from __future__ import annotations

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
