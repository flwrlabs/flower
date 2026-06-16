import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets import load_dataset
from opacus import PrivacyEngine
from torch.utils.data import DataLoader

from flwr.app import Context
from flwr.client import NumPyClient, start_client
from flwr.clientapp import ClientApp

# Define parameters.
PARAMS = {
    "batch_size": 32,
    "train_split": 0.7,
    "local_epochs": 1,
}
PRIVACY_PARAMS = {
    # 'target_epsilon': 5.0,
    "target_delta": 1e-5,
    "noise_multiplier": 0.4,
    "max_grad_norm": 1.2,
}
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Cifar10Dataset(torch.utils.data.Dataset):
    """CIFAR-10 dataset loaded from Hugging Face."""

    def __init__(self, split, transform):
        self.dataset = load_dataset("uoft-cs/cifar10", split=split)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self.transform(item["img"]), item["label"]


# Define model used for training.
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(net, trainloader, privacy_engine, optimizer, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
    epsilon = privacy_engine.get_epsilon(delta=PRIVACY_PARAMS["target_delta"])
    return epsilon


def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    data = Cifar10Dataset("train", transform=transform)
    split = math.floor(len(data) * 0.01 * PARAMS["train_split"])
    trainset = torch.utils.data.Subset(data, list(range(0, split)))
    testset = torch.utils.data.Subset(
        data, list(range(split, math.floor(len(data) * 0.01)))
    )
    trainloader = DataLoader(trainset, PARAMS["batch_size"])
    testloader = DataLoader(testset, PARAMS["batch_size"])
    sample_rate = PARAMS["batch_size"] / len(trainset)
    return trainloader, testloader, sample_rate


model = Net()
trainloader, testloader, sample_rate = load_data()


# Define Flower client.
class FlowerClient(NumPyClient):
    def __init__(self, model) -> None:
        super().__init__()
        # Create a privacy engine which will add DP and keep track of the privacy budget.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=trainloader,
            max_grad_norm=PRIVACY_PARAMS["max_grad_norm"],
            noise_multiplier=PRIVACY_PARAMS["noise_multiplier"],
        )

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        epsilon = train(
            self.model,
            self.trainloader,
            self.privacy_engine,
            self.optimizer,
            PARAMS["local_epochs"],
        )
        print(f"epsilon = {epsilon:.2f}")
        return (
            self.get_parameters(config={}),
            len(self.trainloader),
            {"epsilon": epsilon},
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, testloader)
        return float(loss), len(testloader), {"accuracy": float(accuracy)}


def client_fn(context: Context):
    model = Net()
    return FlowerClient(model).to_client()


app = ClientApp(
    client_fn=client_fn,
)

if __name__ == "__main__":
    start_client(
        server_address="127.0.0.1:8080", client=FlowerClient(model).to_client()
    )
