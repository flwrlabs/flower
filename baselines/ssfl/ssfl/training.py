"""Local training and evaluation matching the legacy SSFL trainer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from ssfl.model import SparseModel, prunable_parameter_names


def train_local(
    model: SparseModel,
    trainloader: DataLoader,
    *,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    max_grad_norm: float,
    round_idx: int,
    lr_scheduler_name: str = "default",
    lr_decay: float = 0.998,
    scheduler_cycle_length: int = 10,
    device: torch.device | None = None,
    masks: dict[str, torch.Tensor] | None = None,
) -> tuple[float, float]:
    """
    Run local SGD for one federated round.

    Returns (average_loss, final_learning_rate).
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)
    if masks:
        model.apply_masks(masks)
    model.train()

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    scheduler = None
    if lr_scheduler_name == "default":
        current_lr = lr * (lr_decay**round_idx)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
    elif lr_scheduler_name in ("cosine-annealing", "cosine_annealing"):
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_cycle_length,
            T_mult=1,
            eta_min=1e-5,
        )
    else:
        raise ValueError(f"Unsupported scheduler: {lr_scheduler_name}")

    total_loss = 0.0
    total_batches = 0
    for _ in range(epochs):
        for batch in trainloader:
            x, labels = _unpack_batch(batch)
            x, labels = x.to(device), labels.to(device)
            model.zero_grad()
            logits = model.forward(x)
            loss = criterion(logits, labels.long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            total_loss += float(loss.item())
            total_batches += 1
        if scheduler is not None:
            scheduler.step()

    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    final_lr = float(optimizer.param_groups[0]["lr"])
    return avg_loss, final_lr


def evaluate_model(
    model: SparseModel,
    testloader: DataLoader,
    *,
    device: torch.device | None = None,
    masks: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)
    if masks:
        model.apply_masks(masks)
    model.eval()

    criterion = nn.CrossEntropyLoss().to(device)
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for batch in testloader:
            x, labels = _unpack_batch(batch)
            x, labels = x.to(device), labels.to(device)
            logits = model(x)
            loss = criterion(logits, labels.long())
            preds = torch.argmax(logits, dim=-1)
            correct += int(preds.eq(labels).sum().item())
            total += int(labels.size(0))
            loss_sum += float(loss.item()) * int(labels.size(0))

    if masks:
        model.remove_pruning()

    return {
        "accuracy": (correct / total) if total > 0 else 0.0,
        "loss": (loss_sum / total) if total > 0 else 0.0,
        "num-examples": float(total),
    }


def count_nonzero_params(state_dict: dict[str, torch.Tensor]) -> int:
    return int(sum(torch.count_nonzero(t).item() for t in state_dict.values()))


def sparsity_from_state_dict(
    state_dict: dict[str, torch.Tensor], model: nn.Module
) -> float:
    prunable = prunable_parameter_names(model)
    total_zeros = 0
    total_weights = 0
    for name in prunable:
        if name not in state_dict:
            continue
        tensor = state_dict[name]
        total_zeros += int((tensor == 0).sum().item())
        total_weights += int(tensor.numel())
    if total_weights == 0:
        return 0.0
    return 100.0 * total_zeros / total_weights


def _unpack_batch(batch) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        # HuggingFace / Flower Datasets batches (CIFAR-10: label; CIFAR-100: fine_label)
        from ssfl.data import labels_from_batch

        if "img" in batch:
            return batch["img"], labels_from_batch(batch)
        if "image" in batch:
            return batch["image"], labels_from_batch(batch)
        raise KeyError(f"Unrecognized batch keys: {list(batch.keys())}")
    return batch[0], batch[1]
