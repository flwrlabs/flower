"""quickstart-pennylane: A Flower / Pennylane Quantum Federated Learning app."""

from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader


def create_quantum_circuit(n_qubits: int):
    """Create quantum device and circuit for the given number of qubits."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def quantum_circuit(inputs, weights):
        """Quantum circuit for the QNN layer."""
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return quantum_circuit


class QuantumNet(nn.Module):
    """Quantum Neural Network combining CNN, classical and quantum layers."""

    def __init__(self, num_classes: int = 10, n_qubits: int = 4, n_layers: int = 3):
        super(QuantumNet, self).__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # CNN feature extraction layers
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Classical dense layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_qubits)

        # Create quantum circuit and layer
        quantum_circuit = create_quantum_circuit(n_qubits)
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qnn = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        # Classical post-processing
        self.fc_out = nn.Linear(n_qubits, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the hybrid CNN-quantum neural network."""
        # CNN feature extraction
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten for dense layers
        x = x.view(-1, 16 * 5 * 5)

        # Classical dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        x = torch.relu(x)

        # Quantum layer
        x = self.qnn(x)

        # Output layer
        x = self.fc_out(x)
        return x


def load_data(
    partition_id: int, num_partitions: int, batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """Load and partition the dataset for federated learning."""

    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="uoft-cs/cifar10",
        partitioners={"train": partitioner},
    )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% validation
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    # Define preprocessing: convert CIFAR-10 images to tensor and normalize pixel values to mean=0.5, std=0.5
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = torch.stack([pytorch_transforms(img) for img in batch["img"]])
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    valloader = DataLoader(
        partition_train_test["test"], batch_size=batch_size, num_workers=0
    )  # validation split
    return trainloader, valloader


def train(
    net: nn.Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> Dict[str, float]:
    """Train the quantum neural network."""
    net.to(device)
    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    running_loss = 0.0
    for _ in range(epochs):
        for batch_idx, batch in enumerate(trainloader):
            data = batch["img"].to(device)
            target = torch.as_tensor(batch["label"], dtype=torch.long, device=device)

            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    # Evaluate on validation set
    val_loss, val_accuracy = test(net, valloader, device)

    avg_train_loss = running_loss / (epochs * len(trainloader))

    return {
        "train_loss": avg_train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    }


def test(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """Evaluate the quantum neural network on validation data."""
    net.to(device)
    net.eval()

    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in testloader:
            data = batch["img"].to(device)
            target = torch.as_tensor(batch["label"], dtype=torch.long, device=device)
            output = net(data)

            test_loss += criterion(output, target).item()

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_loss /= len(testloader)
    accuracy = 100.0 * correct / total

    return test_loss, accuracy
