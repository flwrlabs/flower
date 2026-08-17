"""SSFL saliency score computation: abs(gradient * weight)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ssfl.model import prunable_parameter_names


def screen_gradients(
    model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """Compute per-parameter gradients for a single minibatch (eval mode)."""
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    model.zero_grad()
    x, labels = batch
    x, labels = x.to(device), labels.to(device)
    log_probs = model.forward(x)
    loss = criterion(log_probs, labels.long())
    loss.backward()

    gradients: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.detach().to("cpu")
    return gradients


def calculate_ssfl_scores(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute SSFL saliency scores |gradient * weight| for prunable layers.

    Returns a dict keyed by layer stem names (without ".weight"), matching the
    legacy SSFL implementation.
    """
    if device is None:
        device = next(model.parameters()).device

    grads = screen_gradients(model, batch, device)
    weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    prunable = set(prunable_parameter_names(model))

    saliency_scores: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, weight in weights.items():
            if name in grads and name in prunable:
                stem = name.replace(".weight", "")
                saliency_scores[stem] = torch.abs(grads[name].cpu() * weight)
    return saliency_scores


def average_saliency_over_batches(
    model: nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Average saliency scores over one or more minibatches."""
    if not batches:
        return {}
    total = calculate_ssfl_scores(model, batches[0], device)
    for batch in batches[1:]:
        scores = calculate_ssfl_scores(model, batch, device)
        for key, value in scores.items():
            total[key] = total[key] + value
    n = len(batches)
    return {key: value / n for key, value in total.items()}
