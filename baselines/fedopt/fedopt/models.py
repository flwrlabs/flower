"""ResNet18 models with GroupNorm."""


from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet, resnet18


def get_resnet18_model(num_classes: int = 10) -> nn.Module:
    """Generate ResNet18 model using GroupNormalization rather than BatchNormalization.

    Two groups are used.

    Args:
        num_classes (int, optional): Number of classes {10,100}. Defaults to 10.

    Returns
    -------
        Module: ResNet18 network.
    """
    # ! Contrary to how other FL works use ResNet18 with CIFAR images,
    # This paper made use of ResNet18 without making changes to the arch.
    model: ResNet = resnet18(
        norm_layer=lambda x: nn.GroupNorm(2, x), num_classes=num_classes
    )
    return model


def train(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    device: str,
    learning_rate: float = 0.01,
) -> None:
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    net.train()
    net.to(device)
    for _ in range(epochs):
        for data in trainloader:
            images, labels = data["img"].to(device), data["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def evaluate(
    net: nn.Module, testloader: DataLoader, device: str
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data["img"].to(device), data["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy
