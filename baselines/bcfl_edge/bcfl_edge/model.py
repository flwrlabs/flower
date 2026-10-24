"""bcfl_edge: CNN keyword classifier (Sec. IV-A of the paper)."""

import torch
from torch import nn

from bcfl_edge.dataset import KEYWORDS


class Net(nn.Module):
    """One convolutional block and one dense block, as described in the paper."""

    def __init__(self, num_classes: int = len(KEYWORDS)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),
            nn.ReLU(),
            nn.GroupNorm(32, 64),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 15 * 15, 128),
            nn.ReLU(),
            nn.GroupNorm(32, 128),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of 1x32x32 spectrograms."""
        return self.classifier(self.features(x))


def train(net, trainloader, epochs, learning_rate, device):
    """Train with Adam and sparse categorical cross-entropy."""
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            optimizer.zero_grad()
            outputs = net(batch["spectrogram"].to(device))
            loss = criterion(outputs, batch["label"].to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    return running_loss / (epochs * len(trainloader))


def test(net, testloader, device):
    """Return loss, accuracy and macro-averaged F1 score."""
    net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()
    num_classes = net.classifier[-1].out_features
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    loss = 0.0
    with torch.no_grad():
        for batch in testloader:
            labels = batch["label"].to(device)
            outputs = net(batch["spectrogram"].to(device))
            loss += criterion(outputs, labels).item()
            predicted = outputs.argmax(dim=1)
            confusion += torch.bincount(
                labels.cpu() * num_classes + predicted.cpu(),
                minlength=num_classes**2,
            ).reshape(num_classes, num_classes)
    true_positive = confusion.diag().float()
    precision = true_positive / confusion.sum(dim=0).clamp(min=1)
    recall = true_positive / confusion.sum(dim=1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    accuracy = true_positive.sum().item() / confusion.sum().item()
    return loss / len(testloader), accuracy, f1.mean().item()
