"""Model and deterministic CIFAR-10 partitions shared by both runtimes."""

import hashlib
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


def make_model():
    """Create a ResNet-18 with the standard CIFAR-size input stem."""
    model = models.resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def model_hash(state_dict):
    """Hash all model tensors to detect unchanged global weights."""
    digest = hashlib.sha256()
    for tensor in state_dict.values():
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def load_data(partition_id, num_partitions, batch_size, train):
    """Partition all 50,000 training or 10,000 test examples without overlap."""
    if not 0 <= partition_id < num_partitions:
        raise ValueError("Partition ID is outside the configured cohort")
    operations = []
    if train:
        operations.extend(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    dataset = datasets.CIFAR10(
        root=os.environ["DATA_ROOT"],
        train=train,
        download=False,
        transform=transforms.Compose(operations),
    )
    indices = np.random.default_rng(42).permutation(len(dataset))
    partition = np.array_split(indices, num_partitions)[partition_id].tolist()
    return DataLoader(
        Subset(dataset, partition),
        batch_size=batch_size,
        shuffle=train,
        num_workers=0,
        pin_memory=True,
    )


def execute(model, loader, device, epochs=0, learning_rate=0.01):
    """Train for complete local epochs, or evaluate when epochs is zero."""
    model.to(device)
    model.train(epochs > 0)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    loss_sum, correct, examples, steps = 0.0, 0, 0, 0
    with torch.set_grad_enabled(epochs > 0):
        for _ in range(max(epochs, 1)):
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                if epochs:
                    optimizer.zero_grad()
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
                if epochs:
                    loss.backward()
                    optimizer.step()
                loss_sum += loss.item() * len(labels)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                examples += len(labels)
                steps += 1
    return loss_sum / examples, correct / examples, examples, steps
