"""fedagain: Models and local train / evaluation routines."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18


def build_model(name: str, num_classes: int, in_channels: int) -> nn.Module:
    """Build the model selected in the run config.

    ``resnet18`` is the architecture used in the paper (Sec. 5.1.1). Because the
    benchmark images are small (28x28 or 32x32), the ImageNet stem is replaced by
    a 3x3 stride-1 convolution without max-pooling, the usual CIFAR variant.
    ``cnn`` is a small network useful for quick functional tests.
    """
    if name == "resnet18":
        model = resnet18(weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()  # type: ignore[assignment]
        return model
    if name == "cnn":
        return SmallCNN(num_classes=num_classes, in_channels=in_channels)
    raise ValueError(f"Unknown model '{name}'. Use 'resnet18' or 'cnn'.")


class SmallCNN(nn.Module):
    """A small CNN (two conv blocks with BatchNorm + two linear layers)."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, 10)
        )
        if num_classes != 10:
            self.classifier[-1] = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do forward."""
        return self.classifier(self.features(x))


def train(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    momentum: float,
    device: torch.device,
) -> float:
    """Train the model with SGD and return the mean training loss per sample."""
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    net.train()
    total_loss, total_samples = 0.0, 0
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(net(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def test(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Evaluate the model and return (mean cross-entropy per sample, accuracy)."""
    net.to(device)
    net.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        total_loss += F.cross_entropy(outputs, labels, reduction="sum").item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def benchmark_error(
    net: nn.Module, valloader: DataLoader, device: torch.device
) -> float:
    """Benchmark error E_k of FedAgain (Eq. 6 of the paper).

    Mean cross-entropy of the *received global model* on the local validation set,
    computed in eval mode before any local training step. The loss is averaged
    over samples (not over batches) so that a smaller last batch does not bias it.
    """
    loss, _ = test(net, valloader, device)
    return loss
