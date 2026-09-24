"""Shared utility types and helpers for the GHBM baseline."""

from collections import OrderedDict
from typing import TypeAlias

import torch

StateDict: TypeAlias = OrderedDict[str, torch.Tensor]  # noqa: UP040
MetricScalar = int | float
MetricValue: TypeAlias = MetricScalar | list[int] | list[float]  # noqa: UP040
MetricDict: TypeAlias = dict[str, MetricValue]  # noqa: UP040


def clone_state_dict(state_dict: StateDict) -> StateDict:
    """Clone a state dict onto CPU tensors for safe message/state storage."""
    return OrderedDict(
        (name, tensor.detach().cpu().clone()) for name, tensor in state_dict.items()
    )
