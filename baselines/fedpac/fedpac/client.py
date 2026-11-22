# pylint: disable=too-many-arguments
"""Defines the Flower Client and a function to instantiate it."""


import copy
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import flwr as fl
import torch
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from fedpac.models import fedavg_train, test, train
from fedpac.utils import get_centroid


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
        weight_decay: float,
        momentum: float,
        lamda: float,
    ):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lamda = lamda
        self.num_classes = self.net.num_classes
        self.feature_extractor = self.get_feature_extractor()
        self.feature_centroid = get_centroid(self.feature_extractor)
        self.class_sizes = self.get_class_sizes()
        self.class_fractions = self.get_class_fractions()
        # print(self.class_sizes)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Change the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def get_class_sizes(self):
        dataloader = self.trainloader
        sizes = torch.zeros(self.num_classes)
        for _images, labels in dataloader:
            for i in range(self.num_classes):
                sizes[i] = sizes[i] + (i == labels).sum()
        return sizes

    def get_class_fractions(self):
        total = len(self.trainloader.dataset)
        return self.class_sizes / total

    def get_statistics(self):
        dim = self.net.state_dict()[self.net.classifier_layers[0]][0].shape[0]
        feat_dict = self.get_feature_extractor()
        for k in feat_dict.keys():
            feat_dict[k] = torch.stack(feat_dict[k])

        py = self.class_fractions
        py2 = torch.square(py)
        v = 0
        h_ref = torch.zeros((self.num_classes, dim), device=self.device)
        datasize = torch.tensor(len(self.trainloader)).to(self.device)
        for k in feat_dict.keys():
            feat_k = feat_dict[k]
            num_k = feat_k.shape[0]
            feat_k_mu = feat_k.mean(dim=0)
            h_ref[k] = py[k] * feat_k_mu
            v += (
                py[k] * torch.trace((torch.mm(torch.t(feat_k), feat_k) / num_k))
            ).item()
            v -= (py2[k] * (torch.mul(feat_k_mu, feat_k_mu))).sum().item()
        v = v / datasize.item()
        return (v, h_ref)

    def get_feature_extractor(self):
        """Extract feature extractor layers."""
        feature_extractors = {}
        model = self.net
        train_data = self.trainloader
        with torch.no_grad():
            for inputs, labels in train_data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                features, outputs = model(inputs)
                feature_extractor = features.clone().detach().cpu()
                for i in range(len(labels)):
                    if labels[i].item() not in feature_extractors.keys():
                        feature_extractors[labels[i].item()] = []
                        feature_extractors[labels[i].item()].append(
                            feature_extractor[i, :]
                        )
                    else:
                        feature_extractors[labels[i].item()] = [feature_extractor[i, :]]

        return feature_extractors

    def get_classifier_head(self):
        w = copy.deepcopy(self.net.state_dict())
        keys = self.net.classifier_layers
        for k in keys:
            w[k] = torch.zeros_like(w[k])

        w0 = 0
        for i in range(len(self.avg_head)):
            w0 += self.avg_head[i]
            for k in keys:
                w[k] += self.avg_head[i] * self.net.state_dict()[k]

        for k in keys:
            w[k] = torch.div(w[k], w0)

        return w

    def update_classifier(self, classifier):
        local_weight = self.net.state_dict()
        classifier_keys = self.net.classifier_layers
        for k in local_weight.keys():
            if k in classifier_keys:
                local_weight[k] = classifier[k]
        self.net.load_state_dict(local_weight)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        self.set_parameters(parameters)
        self.stats = self.get_statistics()
        self.global_centroid = config["global_centroid"]
        self.avg_head = config["classifier_head"]
        if self.avg_head is not None:
            classifier_head = self.get_classifier_head()
            self.update_classifier(classifier_head)

        train(
            self.net,
            self.trainloader,
            self.valloader,
            self.num_epochs,
            self.learning_rate,
            self.weight_decay,
            self.momentum,
            self.device,
            self.global_centroid,
            self.feature_centroid,
            self.lamda,
        )
        return (
            self.get_parameters({}),
            len(self.trainloader),
            {
                "centroid": self.feature_centroid,
                "class_sizes": self.class_sizes,
                "stats": (self.stats),
            },
        )

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implement distributed evaluation for a given client."""
        self.set_parameters(parameters)
        self.avg_head = config["classifier_head"]
        if self.avg_head is not None:
            classifier_head = self.get_classifier_head()
            self.update_classifier(classifier_head)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


class FedAvgClient(
    fl.client.NumPyClient
):  # pylint: disable=too-many-instance-attributes
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        weight_decay: float,
        momentum: float,
    ):  # pylint: disable=too-many-arguments
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Change the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        self.set_parameters(parameters)

        fedavg_train(
            self.net,
            self.trainloader,
            self.valloader,
            self.num_epochs,
            self.learning_rate,
            self.weight_decay,
            self.momentum,
            self.device,
        )

        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implement distributed evaluation for a given client."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def gen_client_fn(
    num_epochs: int,
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    learning_rate: float,
    weight_decay: float,
    momentum: float,
    model: DictConfig,
    lamda: float,
) -> Tuple[Callable[[str], FlowerClient], DataLoader]:
    """Generate the client function that creates the Flower Clients.

    Parameters
    ----------
    device : torch.device
        The device on which the the client will train on and test on.
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

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""
        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = model.to(device)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        # Create a  single Flower client representing a single organization
        return FlowerClient(
            net,
            trainloader,
            valloader,
            device,
            num_epochs,
            learning_rate,
            weight_decay,
            momentum,
            lamda,
        )

    return client_fn


def gen_fedavg_client_fn(
    num_epochs: int,
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    learning_rate: float,
    weight_decay: float,
    momentum: float,
    model: DictConfig,
) -> Tuple[Callable[[str], FlowerClient], DataLoader]:
    """Generate the client function that creates the Flower Clients.

    Parameters
    ----------
    device : torch.device
        The device on which the the client will train on and test on.
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

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""
        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = model.to(device)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        # Create a  single Flower client representing a single organization
        return FedAvgClient(
            net,
            trainloader,
            valloader,
            device,
            num_epochs,
            learning_rate,
            weight_decay,
            momentum,
        )

    return client_fn
