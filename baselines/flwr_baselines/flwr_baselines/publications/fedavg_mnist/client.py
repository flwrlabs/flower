# pylint: disable=too-many-arguments
"""Defines the MNIST Flower Client and a function to instantiate it."""


from collections import OrderedDict
from typing import Callable, Dict, Tuple

import flwr as fl
import torch
from flwr.client.client import Client
from flwr.common import Context
from flwr.common.typing import NDArrays, Scalar
from torch.utils.data import DataLoader

from flwr_baselines.publications.fedavg_mnist import model
from flwr_baselines.publications.fedavg_mnist.dataset import load_datasets


class FlowerClient(fl.client.NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
    ):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Returns the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Changes the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implements distributed fit function for a given client."""
        self.set_parameters(parameters)
        model.train(
            self.net,
            self.trainloader,
            self.device,
            epochs=self.num_epochs,
            learning_rate=self.learning_rate,
        )
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implements distributed evaluation for a given client."""
        self.set_parameters(parameters)
        loss, accuracy = model.test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def gen_client_fn(
    device: torch.device,
    iid: bool,
    balance: bool,
    num_clients: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
) -> Tuple[Callable[[Context], Client], DataLoader]:
    """Generates the client function that creates the Flower Clients.

    Parameters
    ----------
    device : torch.device
        The device on which the client will train on and test on.
    iid : bool
        The way to partition the data for each client, i.e. whether the data
        should be independent and identically distributed between the clients
        or if the data should first be sorted by labels and distributed by chunks
        to each client (used to test the convergence in a worst case scenario)
    balance : bool
        Whether the dataset should contain an equal number of samples in each class,
        by default True
    num_clients : int
        The number of clients present in the setup
    num_epochs : int
        The number of local epochs each client should run the training for before
        sending it to the server.
    batch_size : int
        The size of the local batches each client trains on.
    learning_rate : float
        The learning rate for the SGD  optimizer of clients.

    Returns
    -------
    Tuple[Callable[[str], FlowerClient], DataLoader]
        A tuple containing the client function that creates Flower Clients and
        the DataLoader that will be used for testing
    """
    trainloaders, valloaders, testloader = load_datasets(
        iid=iid, balance=balance, num_clients=num_clients, batch_size=batch_size
    )

    def client_fn(context: Context) -> Client:
        """Create a Flower client representing a single organization."""

        # Load model
        net = model.Net().to(device)

        partition_id = context.node_config["partition-id"]

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(partition_id)]
        valloader = valloaders[int(partition_id)]

        # Create a  single Flower client representing a single organization
        return FlowerClient(
            net, trainloader, valloader, device, num_epochs, learning_rate
        ).to_client()

    return client_fn, testloader
