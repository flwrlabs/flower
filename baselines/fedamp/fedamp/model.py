"""FedAMP: Personalized Cross-Silo Federated Learning on Non-IID Data."""

import torch
import torch.nn.functional as F
from torch import nn


class FashionNet(nn.Module):
    """Model for Fashion MNIST."""

    def __init__(self, n_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """Do forward."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')."""

    def __init__(self, n_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """Do forward."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(net, global_net, trainloader, epochs, proximal_weight, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available

    global_net.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    # ensure the gradients for aggregated_net are all None
    for p in global_net.parameters():
        p.requires_grad_(False)

    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()

            # add proximal term (this is the personalization factor)
            proximal_term = 0.0
            for local_weights, global_weights in zip(
                net.parameters(), global_net.parameters(), strict=True
            ):
                proximal_term += torch.sum(
                    (local_weights - global_weights.detach()) ** 2
                )

            loss = criterion(net(images.to(device)), labels.to(device)) + (
                0.5 * proximal_weight * proximal_term
            )

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / (
        len(trainloader) * epochs
    )  # divide by total number of steps
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
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
