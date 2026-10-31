"""Defines the MNIST Flower Client and a function to instantiate it."""

# pylint: disable=too-many-instance-attributes,too-many-arguments
import copy
import os
import pickle
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import flwr as fl
import torch
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from tamuna.models import fedavg_train, tamuna_train
from tamuna.utils import apply_nn_compression


class TamunaClient(fl.client.NumPyClient):
    """Tamuna client for CNN training."""

    STATE_DIR = "client_states"

    def __init__(
        self,
        net: torch.nn.Module,
        trainloader: DataLoader,
        device: torch.device,
        learning_rate: float,
        cid: str,
    ):
        self.net = net
        self.trainloader = trainloader
        self.device = device
        self.learning_rate = learning_rate
        self.control_variate = None
        self.old_compression_mask = None
        self.old_compressed_net = None
        self.cid = cid

        self.state_file_name = f"{TamunaClient.STATE_DIR}/{self.cid}_state.bin"
        self.__create_state()

    def __create_state(self):
        """Create client state."""
        if not os.path.exists(self.state_file_name):
            with open(self.state_file_name, "wb") as handle:
                state = (
                    self.control_variate,
                    self.old_compression_mask,
                    self.old_compressed_net,
                )
                pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Change the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Distributed fit function for a given client."""
        self.set_parameters(parameters)

        mask = config["mask"]
        if isinstance(mask, torch.Tensor):
            mask = mask.to(self.device)
        else:
            raise TypeError("Mask must be a Tensor.")

        self.__load_state()

        self.net, self.control_variate = tamuna_train(
            self.net,
            self.trainloader,
            self.device,
            epochs=int(config["epochs"]),
            eta=float(config["eta"]),
            control_variate=self.control_variate,
            old_compression_mask=self.old_compression_mask,
            old_compressed_net=self.old_compressed_net,
            server_net=copy.deepcopy(self.net),
            learning_rate=self.learning_rate,
        )

        self.net = apply_nn_compression(self.net, mask)

        self.__save_state(mask)

        return self.get_parameters({}), len(self.trainloader), {}

    def __save_state(self, mask):
        """Save client state."""
        with open(self.state_file_name, "wb") as handle:
            state = (
                self.control_variate,
                mask,
                self.net,
            )
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __load_state(self):
        """Load client state."""
        with open(self.state_file_name, "rb") as handle:
            state = pickle.load(handle)
            (
                self.control_variate,
                self.old_compression_mask,
                self.old_compressed_net,
            ) = state


class FedAvgClient(fl.client.NumPyClient):
    """FedAvg client for CNN training."""

    def __init__(
        self,
        net: torch.nn.Module,
        trainloader: DataLoader,
        device: torch.device,
        learning_rate: float,
        cid: int,
    ) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.device = device
        self.learning_rate = learning_rate
        self.cid = cid
        self.net = net

    def set_parameters(self, parameters):
        """Change the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Return the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def fit(self, parameters, config):
        """Distributed fit function for a given client."""
        self.set_parameters(parameters)
        self.net = fedavg_train(
            self.net,
            self.trainloader,
            epochs=config["epochs"],
            learning_rate=self.learning_rate,
            device=self.device,
        )
        return self.get_parameters({}), len(self.trainloader), {}


def gen_tamuna_client_fn(
    trainloaders: List[DataLoader],
    learning_rate: float,
    model: DictConfig,
) -> Callable[[str], TamunaClient]:
    """Generate the client function that creates Tamuna clients.

    Parameters
    ----------
    trainloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset training partition
        belonging to a particular client.
    learning_rate : float
        The learning rate for Tamuna optimizer for clients.
    model: DictConfig
        Architecture of the model being instantiated

    Returns
    -------
    Callable[[str], FlowerClient]
        A tuple containing the client function that creates Tamuna clients
    """

    def tamuna_client_fn(cid: str) -> TamunaClient:
        """Create a Tamuna client."""
        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = instantiate(model).to(device)

        # Note: each client gets a different trainloader, so each client
        # will train on their own unique data
        trainloader = trainloaders[int(cid)]

        return TamunaClient(net, trainloader, device, learning_rate, cid)

    return tamuna_client_fn


def gen_fedavg_client_fn(
    trainloaders: List[DataLoader],
    learning_rate: float,
    model: DictConfig,
) -> Callable[[str], FedAvgClient]:
    """Generate the client function that creates FedAvg clients.

    Parameters
    ----------
    trainloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset training partition
        belonging to a particular client.
    learning_rate : float
        The learning rate for the SGD optimizer for clients.
    model: DictConfig
        Architecture of the model being instantiated

    Returns
    -------
    Callable[[str], FlowerClient]
        A tuple containing the client function that creates FedAvg clients
    """

    def fedavg_client_fn(cid: str) -> FedAvgClient:
        """Create a FedAvg client."""
        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = instantiate(model).to(device)

        # Note: each client gets a different trainloader, so each client
        # will train on their own unique data
        trainloader = trainloaders[int(cid)]

        return FedAvgClient(
            net, trainloader, device, learning_rate=learning_rate, cid=int(cid)
        )

    return fedavg_client_fn
