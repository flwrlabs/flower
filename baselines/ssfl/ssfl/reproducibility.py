"""Deterministic seeding helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def client_round_seed(global_seed: int, client_id: int, round_idx: int) -> int:
    """Derive a deterministic per-client/per-round seed."""
    return int(global_seed) + 1_000_003 * int(client_id) + 97 * int(round_idx)
