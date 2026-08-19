"""Implementation of the model used for the FEMNIST experiments."""

from logging import INFO
from typing import Optional, Tuple

import torch
import torch.nn as nn
from flwr.common.logger import log
from torch import Tensor
from torch.utils.data import DataLoader


class Net(nn.Module):
    """Implementation of the model used in the LEAF paper for training on
    FEMNIST data."""

    # pylint: disable=too-many-instance-attributes
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding="same")
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding="same")
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 2048)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Forward step in training."""
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 7 * 7 * 64)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


# pylint: disable=too-many-arguments, too-many-locals, too-many-branches
def train(
    net: nn.Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    epochs: Optional[int],
    learning_rate: float,
    device: torch.device,
    n_batches: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[float, float, Optional[float], Optional[float]]:
    """Train a given model with CrossEntropy and SGD.

    Alternatively, some version of it, like batch-SGD

    n_batches is an alternative way of specifying the training length
    (instead of epochs)
    """
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    net.train()
    epoch_loss, epoch_acc = 0.0, 0.0
    # pylint: disable=no-else-return
    if epochs:
        for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for images, labels in trainloader:
                correct, epoch_loss, total = train_step(
                    correct,
                    criterion,
                    device,
                    epoch_loss,
                    images,
                    labels,
                    net,
                    optimizer,
                    total,
                )
            epoch_loss = epoch_loss / total
            epoch_acc = correct / total

            if verbose:
                log(
                    INFO,
                    "Epoch %s: train loss %s, accuracy %s",
                    str(epoch + 1),
                    str(epoch_loss),
                    str(epoch_acc),
                )
        # Train loss reported is typically the last epoch loss
        train_loss, train_acc = epoch_loss, epoch_acc
        if len(valloader):
            val_loss, val_acc = test(net, valloader, device)
        else:
            val_loss, val_acc = None, None
        return train_loss, train_acc, val_loss, val_acc
    elif n_batches:
        # Training time given in number of batches not epochs
        correct, total, train_loss = 0, 0, 0.0
        for batch_idx, (images, labels) in enumerate(trainloader):
            if batch_idx == n_batches:
                break
            correct, train_loss, total = train_step(
                correct,
                criterion,
                device,
                train_loss,
                images,
                labels,
                net,
                optimizer,
                total,
            )
        train_acc = correct / total
        train_loss = train_loss / total
        if verbose:
            log(
                INFO,
                "Batch len based training: train loss %s, accuracy %s",
                str(train_loss),
                str(train_acc),
            )
        if len(valloader):
            val_loss, val_acc = test(net, valloader, device)
        else:
            val_loss, val_acc = None, None
        return train_loss, train_acc, val_loss, val_acc
    else:
        raise ValueError("either n_epochs or n_batches should be specified ")


def train_step(
    correct: int,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    epoch_loss: float,
    images: Tensor,
    labels: Tensor,
    net: nn.Module,
    optimizer: torch.optim.SGD,
    total: int,
) -> Tuple[int, float, int]:
    """Single train step.

    Returns
    -------
    correct, epoch_loss, total: Tuple[int, float, int]
        number of correctly predicted samples, sum of loss, total number of
        samples
    """
    images = images.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    outputs = net(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
    total += labels.size(0)
    correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return correct, float(epoch_loss), total


def test(
    net: nn.Module, dataloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """Test - calculate metrics on the given dataloader."""
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    if len(dataloader) == 0:
        raise ValueError("Dataloader can't be 0, exiting...")
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            output = net(images)
            loss += criterion(output, labels).item()
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        accuracy = correct / total
        loss /= total
    return float(loss), accuracy
