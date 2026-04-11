"""Local training and evaluation utilities."""

import torch
from torch import nn
from torch.utils.data import DataLoader

from ghbm.algorithm import TrainingModifiers


def train(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    device: torch.device,
    *,
    learning_rate: float = 0.1,
    weight_decay: float = 0.0,
    beta: float = 0.0,
    modifiers: TrainingModifiers | None = None,
) -> float:
    """Train the model on the training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=learning_rate,
        momentum=0.0,
        weight_decay=weight_decay,
    )
    net.train()
    running_loss = 0.0
    modifiers = modifiers or TrainingModifiers()
    total_steps = max(1, epochs * len(trainloader))
    momentum_scale = beta / (learning_rate * total_steps)
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            _apply_training_modifiers(
                net=net,
                modifiers=modifiers,
                beta=beta,
                device=device,
                momentum_scale=momentum_scale,
            )
            optimizer.step()
            running_loss += loss.item()

    return running_loss / total_steps


def test(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return loss / len(testloader), correct / len(testloader.dataset)


def _apply_training_modifiers(
    net: nn.Module,
    modifiers: TrainingModifiers,
    beta: float,
    device: torch.device,
    momentum_scale: float,
) -> None:
    """Apply the algorithm-specific gradient correction before optimizer step."""
    if modifiers.anchor_model is not None and beta != 0.0:
        for name, parameter in net.named_parameters():
            if parameter.grad is None:
                continue
            correction = modifiers.anchor_model[name].to(
                device=device, dtype=parameter.grad.dtype
            )
            correction = correction - parameter.detach()
            parameter.grad.add_(
                correction,
                alpha=momentum_scale * modifiers.anchor_scale,
            )
        return

    if modifiers.server_momentum is not None and beta != 0.0:
        for name, parameter in net.named_parameters():
            if parameter.grad is None:
                continue
            correction = modifiers.server_momentum[name].to(
                device=device, dtype=parameter.grad.dtype
            )
            parameter.grad.add_(correction, alpha=momentum_scale)
