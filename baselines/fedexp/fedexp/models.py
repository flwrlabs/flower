"""Model definitions for FedExp."""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from flwr.common import Scalar
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.GroupNorm(2, planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.GroupNorm(2, planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(2, self.expansion * planes),
            )

    def forward(self, x):
        """Forward pass through the block."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class _ResNet(nn.Module):
    def __init__(self, num_classes, num_blocks, block=_BasicBlock):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(2, 64)
        self.layers = nn.ModuleList()
        for planes, num_block, stride in zip([64, 128, 256, 512], num_blocks, [1, 2, 2, 2]):
            strides = [stride] + [1] * (num_block - 1)
            block_layers = []
            for i_stride in strides:
                block_layers.append(block(self.in_planes, planes, i_stride))
                self.in_planes = planes * block.expansion
            self.layers.extend(block_layers)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        """Forward pass through the network."""
        out = F.relu(self.bn1(self.conv1(x)))
        for layer in self.layers:
            out = layer(out)
        out = F.avg_pool2d(out, 3)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet18(_ResNet):
    """ResNet18 model."""

    def __init__(self, num_classes=10):
        super().__init__(num_classes=num_classes, num_blocks=[2, 2, 2, 2])


def test(
        net: nn.Module, test_loader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """Evaluate the network on the entire test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to test.
    test_loader : DataLoader
        The DataLoader containing the data to test the network on.
    device : torch.device
        The device on which the model should be tested, either 'cpu' or 'cuda'.

    Returns
    -------
    Tuple[float, float]
        The loss and the accuracy of the input model on the given data.
    """
    if len(test_loader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")

    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, "Testing ..."):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(test_loader.dataset)
    accuracy = correct / total
    return loss, accuracy


def train(  # pylint: disable=too-many-arguments
        net: nn.Module,
        trainloader: DataLoader,
        device: torch.device,
        epochs: int,
        hyperparams: Dict[str, Scalar],
) -> None:
    """Train the network on the training set.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    device : torch.device
        The device on which the model should be trained, either 'cpu' or 'cuda'.
    epochs : int
        The number of epochs the model should be trained for.
    hyperparams : Dict[str, Scalar]
        The hyperparameters to use for training.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=hyperparams["eta_l"],
        momentum=0,
        weight_decay=hyperparams["weight_decay"],
    )
    net.train()
    counter = 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        if hyperparams["use_data_augmentation"]:
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
            )
            images = transform_train(images)
        net.zero_grad()
        log_probs = net(images)
        loss = criterion(log_probs, labels)
        loss.backward()
        if hyperparams["use_gradient_clipping"]:
            torch.nn.utils.clip_grad_norm_(
                parameters=net.parameters(), max_norm=hyperparams["max_norm"]
            )
        optimizer.step()
        counter += 1
        if counter >= epochs:
            break


if __name__ == "__main__":
    model1 = ResNet18()
    m1_params = model1.parameters()
    m1_vec = parameters_to_vector(m1_params)
    model2 = ResNet18()
    m2_params = model2.parameters()
    m2_vec = parameters_to_vector(m2_params)
    print("Testing")
