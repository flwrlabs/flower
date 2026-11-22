"""fedpac: A Flower Baseline."""

from collections import OrderedDict
from typing import Tuple, Dict, Any

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader


class EMNISTNet(nn.Module):
    """Implementation of the model used in the FedPAC paper for training.

    on EMNIST data.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=0)
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=0)
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(5 * 5 * 32, 128)
        self.relu3 = nn.LeakyReLU()
        self.fc2 = nn.Linear(128, num_classes)
        self.num_classes = num_classes
        self.feature_layers = [
            "conv1.weight",
            "conv1.bias",
            "conv2.weight",
            "conv2.bias",
            "fc1.weight",
            "fc1.bias",
        ]
        self.classifier_layers = ["fc2.weight", "fc2.bias"]

    def forward(self, x: Tensor) -> Tensor:
        """Forward step in training."""
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.relu3(x)
        y = self.fc2(x)
        return x, y

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CIFARNet(nn.Module):
    """Implementation of the model used in the FedPAC paper for training on CIFAR10
    data.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=0)
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=1)
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.LeakyReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(3 * 3 * 64, 128)
        self.relu3 = nn.LeakyReLU()
        self.fc2 = nn.Linear(128, num_classes)
        self.num_classes = num_classes
        self.feature_layers = [
            "conv1.weight",
            "conv1.bias",
            "conv2.weight",
            "conv2.bias",
            "conv3.weight",
            "conv3.bias",
            "fc1.weight",
            "fc1.bias",
        ]
        self.classifier_layers = ["fc2.weight", "fc2.bias"]

    def forward(self, x: Tensor) -> Tensor:
        """Forward step in training."""
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.relu3(x)
        y = self.fc2(x)
        return x, y
        
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
def train(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    device: torch.device,
    learning_rate: float = 0.1,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    global_centroid: Dict = {},
    feature_centroid: Dict = {},
    lamda: float = 0.1,
) -> float:
    """Train the network on the training set."""
    net.train()
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # Phase 1: classifier training
    for name, param in net.named_parameters():
        if name in net.feature_layers:
            param.requires_grad = False
        else:
            param.requires_grad = True

    params = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = torch.optim.SGD(
        params, lr=0.1, weight_decay=weight_decay, momentum=momentum
    )
    net = train_classifier_epoch(net, trainloader, device, criterion, optimizer)

    # Phase 2: feature extractor training
    for name, param in net.named_parameters():
        if name in net.feature_layers:
            param.requires_grad = True
        else:
            param.requires_grad = False

    params = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = torch.optim.SGD(
        params, lr=learning_rate, weight_decay=weight_decay, momentum=momentum
    )
    for _ in range(epochs):
        running_loss += train_features_epoch(
            net,
            trainloader,
            device,
            criterion,
            optimizer,
            global_centroid,
            feature_centroid,
            lamda,
        )

    return running_loss / len(trainloader)

def train_classifier_epoch(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    criterion: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.Adam,
) -> nn.Module:
    """Train classifier for one epoch."""
    net = net.to(device)
    for batch in trainloader:
        images = batch["img"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        _, out = net(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
    return net

def train_features_epoch(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    criterion: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.Adam,
    global_centroid,
    feature_centroid,
    lamda,
) -> nn.Module:
    """Train feature extractor for one epoch."""
    net = net.to(device)
    for batch in trainloader:
        images = batch["img"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        feat, out = net(images)
        feat_new = feat.clone().detach()
        loss0 = criterion(out, labels)
        if global_centroid != {}:
            for i in range(len(labels)):
                if labels[i].item() in global_centroid.keys():
                    feat[i] = global_centroid[labels[i].item()].detach()
                else:
                    feat[i] = feature_centroid[labels[i].item()].detach()
            loss_fn = nn.MSELoss()
            loss1 = loss_fn(feat_new, feat)
        else:
            loss1 = torch.tensor(0)
        loss = loss0 + lamda * loss1
        loss.backward()
        optimizer.step()
    return net


def test(net: nn.Module, testloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    net = net.to(device)
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            features, outputs = net(images)  # Unpack both feature and output
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    """Extract model parameters as numpy arrays from state_dict."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Apply parameters to an existing model."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)