"""Sample-weighted FedAvg matching the legacy SSFL runner."""

from __future__ import annotations

from collections import OrderedDict

import torch


def fedavg_weighted(
    updates: list[tuple[float, dict[str, torch.Tensor]]],
) -> OrderedDict[str, torch.Tensor]:
    """
    Aggregate local state dicts with sample-count weights.

    Args:
        updates: list of (num_examples, state_dict)
    """
    if not updates:
        raise ValueError("Cannot aggregate an empty update list")

    total = float(sum(n for n, _ in updates))
    if total <= 0:
        raise ValueError("Total number of examples must be positive")

    first = updates[0][1]
    device = next(iter(first.values())).device
    sample_weights = torch.tensor(
        [n / total for n, _ in updates], device=device, dtype=torch.float32
    )

    global_model: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in first.keys():
        layer_tensors = torch.stack(
            [state[key].to(device) for _, state in updates], dim=0
        )
        weights = sample_weights.view([-1] + [1] * (layer_tensors.dim() - 1))
        global_model[key] = torch.sum(layer_tensors * weights, dim=0)
    return global_model
