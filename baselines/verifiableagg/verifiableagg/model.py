"""Model and train/eval utilities for verifiable aggregation baseline."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


class Net(nn.Module):
    """Small MLP for deterministic synthetic binary classification."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute model outputs."""
        return self.net(x)


def train(
    model: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> float:
    """Train model for a number of local epochs and return average loss."""
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    loss_sum = 0.0
    num_batches = 0

    for _ in range(epochs):
        for x, y in trainloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            num_batches += 1

    return loss_sum / max(num_batches, 1)


def evaluate(
    model: nn.Module, valloader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Evaluate model and return loss and accuracy."""
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in valloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss_sum += float(criterion(logits, y).item())
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == y).sum().item())
            total += int(y.shape[0])

    avg_loss = loss_sum / max(len(valloader), 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy
