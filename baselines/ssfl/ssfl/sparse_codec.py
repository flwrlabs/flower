"""Static-mask sparse packing for Phase-2 wire transport."""

from __future__ import annotations

from collections import OrderedDict

import torch


def pack_state_dict(
    state_dict: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """
    Pack a model state dict using a static binary mask.

    Masked parameters are stored as 1-D tensors of active values only.
    Non-masked parameters (biases, norms, etc.) remain dense.
    """
    packed: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, tensor in state_dict.items():
        cpu = tensor.detach().cpu().contiguous()
        if name in masks:
            mask = masks[name].detach().cpu().to(torch.bool)
            if mask.shape != cpu.shape:
                raise ValueError(
                    f"Mask shape mismatch for {name}: {tuple(mask.shape)} vs "
                    f"{tuple(cpu.shape)}"
                )
            packed[name] = cpu[mask].contiguous()
        else:
            packed[name] = cpu
    return packed


def unpack_state_dict(
    packed: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """
    Expand a packed state dict back to dense tensors using the static mask.

    Inactive positions are filled with zeros.
    """
    dense: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, values in packed.items():
        values = values.detach().cpu().contiguous()
        if name in masks:
            mask = masks[name].detach().cpu().to(torch.bool)
            expected = int(mask.sum().item())
            if values.numel() != expected:
                raise ValueError(
                    f"Packed length mismatch for {name}: got {values.numel()}, "
                    f"expected {expected} active values"
                )
            full = torch.zeros(mask.shape, dtype=values.dtype)
            full[mask] = values
            dense[name] = full
        else:
            dense[name] = values
    return dense


def packed_numel(packed: dict[str, torch.Tensor]) -> int:
    return int(sum(t.numel() for t in packed.values()))


def dense_numel(state_dict: dict[str, torch.Tensor]) -> int:
    return int(sum(t.numel() for t in state_dict.values()))


def compression_ratio(
    state_dict: dict[str, torch.Tensor],
    packed: dict[str, torch.Tensor],
) -> float:
    dense = dense_numel(state_dict)
    if dense == 0:
        return 1.0
    return packed_numel(packed) / dense
