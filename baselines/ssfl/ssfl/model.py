"""CIFAR ResNet with GroupNorm and SparseModel mask wrapper."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

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
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

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
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, class_num=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, class_num)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for layer_stride in strides:
            layers.append(block(self.in_planes, planes, layer_stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.linear(out)


def _replace_bn_with_gn(model: nn.Module, num_groups: int = 32) -> None:
    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            groups = min(num_groups, module.num_features)
            gn = nn.GroupNorm(groups, module.num_features)
            parts = name.split(".")
            if len(parts) > 1:
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                child_name = parts[-1]
            else:
                parent = model
                child_name = parts[0]
            bn_layers.append((parent, child_name, gn))
    for parent, child_name, gn in bn_layers:
        setattr(parent, child_name, gn)


def customized_resnet18(class_num: int = 10) -> ResNet:
    """ResNet-18 with GroupNorm for federated learning (matches SSFL paper)."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], class_num=class_num)
    model.bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
    model.layer1[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
    model.layer1[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=64)
    model.layer1[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
    model.layer1[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=64)
    model.layer2[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=128)
    model.layer2[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=128)
    model.layer2[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=128)
    model.layer2[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=128)
    model.layer2[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=128)
    model.layer3[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
    model.layer3[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=256)
    model.layer3[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=256)
    model.layer3[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
    model.layer3[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=256)
    model.layer4[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=512)
    model.layer4[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=512)
    model.layer4[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=512)
    model.layer4[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=512)
    model.layer4[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=512)
    return model


def customized_resnet50(class_num: int = 10) -> ResNet:
    """ResNet-50 with GroupNorm for federated learning (matches SSFL paper)."""
    model = ResNet(Bottleneck, [3, 4, 6, 3], class_num=class_num)
    _replace_bn_with_gn(model, num_groups=32)
    return model


class SparseModel(nn.Module):
    """Wrapper that applies binary masks via torch pruning and flattens state_dicts."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.add_module("model", model)
        self.pruned_layers: set[str] = set()

    def forward(self, x):
        return self._modules["model"].forward(x)

    def apply_masks(self, masks: dict[str, torch.Tensor]) -> None:
        self.remove_pruning()
        for name, module in self._modules["model"].named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                param_name = f"{name}.weight"
                if param_name in masks and masks[param_name] is not None:
                    mask_on_device = masks[param_name].to(module.weight.device)
                    prune.custom_from_mask(module, name="weight", mask=mask_on_device)
                    self.pruned_layers.add(name)

    def remove_pruning(self) -> None:
        for name, module in self._modules["model"].named_modules():
            if name in self.pruned_layers and prune.is_pruned(module):
                prune.remove(module, "weight")
        self.pruned_layers.clear()

    def state_dict(self, *args, **kwargs):
        kwargs.pop("destination", None)
        pruned_state_dict = self._modules["model"].state_dict(*args, **kwargs)
        flat_state_dict = OrderedDict()
        for key, value in pruned_state_dict.items():
            if key.endswith(".weight_mask"):
                continue
            if key.endswith(".weight_orig"):
                base_name = key.rsplit(".", 1)[0]
                mask = pruned_state_dict[f"{base_name}.weight_mask"]
                flat_state_dict[f"{base_name}.weight"] = value * mask
            else:
                flat_state_dict[key] = value
        return flat_state_dict

    def load_state_dict(self, state_dict, strict=True):
        pruned_load_dict = OrderedDict()
        for key, value in state_dict.items():
            base_name = key.rsplit(".", 1)[0]
            if key.endswith(".weight") and base_name in self.pruned_layers:
                pruned_load_dict[f"{base_name}.weight_orig"] = value
            else:
                pruned_load_dict[key] = value
        return self._modules["model"].load_state_dict(pruned_load_dict, strict=False)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return self._modules["model"].named_parameters(prefix=prefix, recurse=recurse)

    def named_modules(self, memo=None, prefix: str = "", remove_duplicate: bool = True):
        return self._modules["model"].named_modules(
            memo=memo, prefix=prefix, remove_duplicate=remove_duplicate
        )


def create_model(model_name: str, num_classes: int) -> SparseModel:
    if model_name == "resnet18":
        backbone = customized_resnet18(class_num=num_classes)
    elif model_name == "resnet50":
        backbone = customized_resnet50(class_num=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return SparseModel(backbone)


def prunable_parameter_names(model: nn.Module) -> list[str]:
    names = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            names.append(f"{name}.weight")
    return names


def num_classes_for_dataset(dataset_name: str) -> int:
    if dataset_name == "cifar10":
        return 10
    if dataset_name == "cifar100":
        return 100
    raise ValueError(f"Unsupported dataset: {dataset_name}")
