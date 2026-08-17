"""Global mask creation from aggregated saliency scores."""

from __future__ import annotations

import hashlib
from collections import OrderedDict

import torch


def get_mean_saliency_scores(
    scores_gathered: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Uniform (unweighted) mean of per-client saliency score dictionaries."""
    if not scores_gathered:
        return {}

    avg_scores = {k: v.clone().detach() for k, v in scores_gathered[0].items()}
    for score_dict in scores_gathered[1:]:
        for key, value in score_dict.items():
            if key in avg_scores:
                avg_scores[key] = avg_scores[key] + value
            else:
                avg_scores[key] = value.clone().detach()

    n = len(scores_gathered)
    for key in avg_scores:
        avg_scores[key] = avg_scores[key] / n
    return avg_scores


def get_weighted_mean_saliency_scores(
    scores_gathered: list[dict[str, torch.Tensor]],
    weights: list[float],
) -> dict[str, torch.Tensor]:
    """Sample-count-weighted mean of saliency scores (opt-in)."""
    if not scores_gathered:
        return {}
    if len(scores_gathered) != len(weights):
        raise ValueError("scores_gathered and weights must have the same length")

    total_w = float(sum(weights))
    if total_w <= 0:
        raise ValueError("sum of weights must be positive")

    avg_scores = {
        k: v.clone().detach() * (weights[0] / total_w)
        for k, v in scores_gathered[0].items()
    }
    for score_dict, weight in zip(scores_gathered[1:], weights[1:]):
        scale = weight / total_w
        for key, value in score_dict.items():
            contrib = value * scale
            if key in avg_scores:
                avg_scores[key] = avg_scores[key] + contrib
            else:
                avg_scores[key] = contrib.clone().detach()
    return avg_scores


def create_mask_from_scores(
    scores_dict: dict[str, torch.Tensor],
    keep_ratio: float,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """
    Create a binary mask by keeping scores >= the k-th largest score.

    Ties at the threshold are retained (legacy SSFL behavior), so the resulting
    density can slightly exceed keep_ratio.
    """
    if not scores_dict:
        raise ValueError("The saliency score dictionary is empty.")

    all_scores = torch.cat([v.flatten() for v in scores_dict.values()])
    num_params_to_keep = int(len(all_scores) * keep_ratio)
    if num_params_to_keep < 1:
        threshold = float("inf")
    else:
        threshold = torch.topk(all_scores, num_params_to_keep, sorted=True).values[-1]

    final_weight_mask: dict[str, torch.Tensor] = {}
    layer_wise_density: dict[str, float] = {}
    with torch.no_grad():
        for name, scores in scores_dict.items():
            mask = (scores >= threshold).float().to(device)
            param_name = f"{name}.weight"
            final_weight_mask[param_name] = mask
            layer_wise_density[param_name] = float(mask.mean().item())
    return final_weight_mask, layer_wise_density


def apply_mask_to_state_dict(
    state_dict: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Zero out prunable weights according to the global mask."""
    masked = OrderedDict()
    for name, tensor in state_dict.items():
        if name in masks:
            masked[name] = tensor * masks[name].to(tensor.device)
        else:
            masked[name] = tensor.clone()
    return masked


def mask_digest(masks: dict[str, torch.Tensor]) -> str:
    """Stable hash of a mask dictionary for version checks."""
    hasher = hashlib.sha256()
    for key in sorted(masks.keys()):
        hasher.update(key.encode("utf-8"))
        arr = masks[key].detach().cpu().contiguous().numpy().tobytes()
        hasher.update(arr)
    return hasher.hexdigest()[:16]


def masks_to_cpu_uint8(masks: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().to(torch.uint8) for k, v in masks.items()}


def masks_from_uint8(masks: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().to(torch.float32) for k, v in masks.items()}
