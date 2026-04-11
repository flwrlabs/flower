"""Model definitions for the GHBM baseline."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init

from ghbm.config import ModelName, NormLayer


class LeNet(nn.Module):
    """LeNet variant from the original GHBM codebase."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, num_classes)

    def forward(self, x):
        """Run forward pass."""
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def _weights_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear | nn.Conv2d):
        init.kaiming_normal_(module.weight)


class LambdaLayer(nn.Module):
    """Wrap a lambda into a torch module."""

    def __init__(self, lambd) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the wrapped lambda."""
        return self.lambd(x)


class BasicBlock(nn.Module):
    """CIFAR ResNet basic block from the original GHBM codebase."""

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        norm_layer: NormLayer = NormLayer.GROUP,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = (
            nn.GroupNorm(2, planes)
            if norm_layer is NormLayer.GROUP
            else nn.BatchNorm2d(planes)
        )
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = (
            nn.GroupNorm(2, planes)
            if norm_layer is NormLayer.GROUP
            else nn.BatchNorm2d(planes)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(
                lambda x: F.pad(  # pylint: disable=not-callable
                    x[:, :, ::2, ::2],
                    (0, 0, 0, 0, planes // 4, planes // 4),
                    "constant",
                    0,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    """CIFAR ResNet from the original GHBM codebase."""

    _num_blocks = {
        20: [3, 3, 3],
        32: [5, 5, 5],
        44: [7, 7, 7],
        56: [9, 9, 9],
        110: [18, 18, 18],
        202: [200, 200, 200],
    }

    def __init__(
        self,
        num_classes: int,
        version: int = 20,
        norm_layer: NormLayer = NormLayer.GROUP,
    ) -> None:
        super().__init__()
        if version not in self._num_blocks:
            raise ValueError(
                f"Unknown ResNet version {version}. "
                f"Available are {list(self._num_blocks.keys())}"
            )

        num_blocks = self._num_blocks[version]
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = (
            nn.GroupNorm(2, 16) if norm_layer is NormLayer.GROUP else nn.BatchNorm2d(16)
        )
        self.layer1 = self._make_layer(
            16, num_blocks[0], stride=1, norm_layer=norm_layer
        )
        self.layer2 = self._make_layer(
            32, num_blocks[1], stride=2, norm_layer=norm_layer
        )
        self.layer3 = self._make_layer(
            64, num_blocks[2], stride=2, norm_layer=norm_layer
        )
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(
        self,
        planes: int,
        num_blocks: int,
        stride: int,
        norm_layer: NormLayer,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for current_stride in strides:
            layers.append(
                BasicBlock(self.in_planes, planes, current_stride, norm_layer)
            )
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size(3))  # pylint: disable=not-callable
        out = out.view(out.size(0), -1)
        return self.linear(out)


def create_model(
    model_name: ModelName,
    num_classes: int,
    resnet_version: int = 20,
    norm_layer: NormLayer = NormLayer.GROUP,
) -> nn.Module:
    """Construct a model from run-config values."""
    if model_name is ModelName.LENET:
        return LeNet(num_classes=num_classes)
    if model_name is ModelName.RESNET:
        return ResNet(
            num_classes=num_classes,
            version=resnet_version,
            norm_layer=norm_layer,
        )
    raise ValueError(f"Unsupported model: {model_name}")
