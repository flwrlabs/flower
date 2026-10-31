"""Defines the MNIST Flower Client and a function to instantiate it."""

from collections import OrderedDict
from typing import Dict

import flwr as fl
import torch
from flwr.common import NDArrays, Scalar
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader

from fedntd.models import Net, apply_transforms, test, train


class FlowerClient(fl.client.NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(self, trainloader, valloader) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader
        self.model = Net(num_classes=10)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def set_parameters(self, parameters):
        """Change the parameters of the model using the given ones."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Return the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Implement distributed fit function for a given client."""
        self.set_parameters(parameters)
        lr, epochs = config["lr"], config["epochs"]
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        train(
            self.model,
            self.trainloader,
            optim,
            epochs=epochs,
            device=self.device,
            tau=1.0,
            beta=1.0,
            num_classes=10,
        )
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Implement distributed evaluation for a given client."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, device=self.device)
        return float(loss), len(self.valloader), {"accuracy": accuracy}


def get_client_fn(dataset: FederatedDataset):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""
        client_dataset = dataset.load_partition(int(cid), "train")
        client_dataset_splits = client_dataset.train_test_split(test_size=0.1)
        trainset = client_dataset_splits["train"]
        valset = client_dataset_splits["test"]
        trainloader = DataLoader(
            trainset.with_transform(apply_transforms), batch_size=32, shuffle=True
        )
        valloader = DataLoader(valset.with_transform(apply_transforms), batch_size=32)
        return FlowerClient(trainloader, valloader).to_client()

    return client_fn
