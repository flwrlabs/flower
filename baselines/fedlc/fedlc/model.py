"""fedlc: A Flower Baseline."""

from collections import OrderedDict

import torch
import torch.nn as nn


class CNNModel(nn.Module):
    """CNN model as described in Appendix of FedLC paper."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, 3)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        """Forward pass."""
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.max_pool2d(x, 2, stride=2)

        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.max_pool2d(x, 2, stride=2)

        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = nn.functional.max_pool2d(x, 2, stride=2)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


def train(net, trainloader, epochs, device, learning_rate, criterion, proximal_mu):
    """Train the model on the training set."""
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    net.train()
    running_loss = 0.0

    global_params = []
    if proximal_mu > 0:
        global_params = [param.detach().clone() for param in net.parameters()]

    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            logits = net(images.to(device))

            proximal_term = 0.0
            for local_weights, global_weights in zip(net.parameters(), global_params):
                proximal_term += (local_weights - global_weights).norm(2)


            loss = criterion(logits, labels.to(device))
            loss += (proximal_mu / 2) * proximal_term
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    net.eval()
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


def get_parameters(net):
    """Extract model parameters as numpy arrays from state_dict."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters):
    """Apply parameters to an existing model."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
