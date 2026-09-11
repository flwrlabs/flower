"""floco: A Flower Baseline."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
    """A simple CNN model for CIFAR-10."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=0, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=1, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        """Compute forward pass through the model."""
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = F.leaky_relu(self.fc1(x))
        return self.classifier(x)


class SimplexModel(torch.nn.Module):
    """A wrapper that creates a PyTorch model with a simplex classifier."""

    def __init__(self, endpoints) -> None:
        super().__init__()
        self.endpoints = endpoints
        self.base_model = Net()
        self.base_model.classifier = _SimplexLinear(
            endpoints=self.endpoints,
            in_features=self.base_model.classifier.in_features,
            out_features=self.base_model.classifier.out_features,
            bias=True,
        )
        self.subregion_parameters: tuple[np.ndarray, float] | None = None
        self.training: bool = True

    def forward(self, x):
        """Compute forward pass through the model."""
        if self.subregion_parameters is None:  # before projection
            if self.training:  # sample uniformly from simplex for training
                sample = np.random.exponential(scale=1.0, size=self.endpoints)
                self.base_model.classifier.alphas = tuple(sample / sample.sum())
            else:  # use simplex center for testing
                simplex_center = [1 / self.endpoints for _ in range(self.endpoints)]
                self.base_model.classifier.alphas = simplex_center
        else:  # after projection
            if self.training:  # sample uniformly from subregion for training
                self.base_model.classifier.alphas = tuple(
                    _sample_l1_ball(*self.subregion_parameters)
                )
            else:  # use subregion center for testing
                self.base_model.classifier.alphas = tuple(self.subregion_parameters[0])
        return self.base_model(x)


def create_model(context) -> torch.nn.Module:
    """Create a model based on the algorithm in run config."""
    seed = int(context.run_config["seed"])
    torch.manual_seed(seed)
    algorithm = str(context.run_config["algorithm"])
    if algorithm == "Floco":
        endpoints = int(context.run_config["endpoints"])
        return SimplexModel(endpoints=endpoints)
    if algorithm == "FedAvg":
        return Net()
    raise ValueError(f"Algorithm not implemented: {algorithm}")


def train(net, trainloader, epochs, device, reg_params=None, lamda=0.0):
    """Train the model on the training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=0.02, momentum=0.5, weight_decay=5e-4
    )
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            loss = criterion(net(images), labels)
            optimizer.zero_grad()
            loss.backward()
            if reg_params is not None:
                _regularize_pers_model(net, reg_params, lamda)
            optimizer.step()
            running_loss += loss.item()
    return running_loss / len(trainloader)


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


class _SimplexLinear(torch.nn.Linear):
    """A fully-connected simplex classifier layer."""

    def __init__(self, endpoints: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.endpoints = endpoints
        self.alphas = [1 / endpoints for _ in range(endpoints)]
        self._weights = torch.nn.ParameterList(
            [_initialize_weight(self.weight) for _ in range(endpoints)]
        )

    @property
    def weight(self) -> torch.nn.Parameter:
        """Return the weighted sum of the parameter weights."""
        return sum(alpha * weight for alpha, weight in zip(self.alphas, self._weights))


def _initialize_weight(init_weight: torch.Tensor) -> torch.nn.Parameter:
    """Initialize a weight tensor with Xavier normal initialization."""
    weight = torch.nn.Parameter(torch.zeros_like(init_weight))
    torch.nn.init.xavier_normal_(weight)
    return weight


def _sample_l1_ball(center, radius):
    """Sample uniformly from L1 ball."""
    u = np.random.uniform(-1, 1, len(center))
    u = np.sign(u) * (np.abs(u) / np.sum(np.abs(u)))
    return center + np.random.uniform(0, radius) * u


def _regularize_pers_model(model, reg_model_params, lamda):
    """Regularize the personal model with the global model."""
    for pers_param, global_param in zip(model.parameters(), reg_model_params):
        if pers_param.requires_grad and pers_param.grad is not None:
            pers_param.grad.data += lamda * (pers_param.data - global_param.data)
