import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = [
    "ResNet",
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "resnet110",
    "resnet1202",
]


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A", device=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            device=device,
        )

        num_groups = 2
        self.gn1 = nn.GroupNorm(
            num_groups=num_groups, num_channels=planes, device=device
        )
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            device=device,
        )

        self.gn2 = nn.GroupNorm(
            num_groups=num_groups, num_channels=planes, device=device
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """For CIFAR10 ResNet paper uses option A."""
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                        device=device,
                    ),
                    nn.GroupNorm(
                        num_groups=2,
                        num_channels=self.expansion * planes,
                        device=device,
                    ),
                )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, device=None):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False, device=device
        )

        num_groups = 2
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=16, device=device)
        self.layer1 = self._make_layer(
            block, 16, num_blocks[0], stride=1, device=device
        )
        self.layer2 = self._make_layer(
            block, 32, num_blocks[1], stride=2, device=device
        )
        self.layer3 = self._make_layer(
            block, 64, num_blocks[2], stride=2, device=device
        )
        self.linear = nn.Linear(64, num_classes, device=device)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, device=None):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, device=device))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(params, device=None):
    return ResNet(BasicBlock, [3, 3, 3], device=device)


def resnet32(params):
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44(params):
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56(params):
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110(params):
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202(params):
    return ResNet(BasicBlock, [200, 200, 200])
