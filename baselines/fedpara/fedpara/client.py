"""Client for FedPara."""

import copy
import os
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
import torch
from flwr.common import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from fedpara.models import test, train
from fedpara.utils import get_keys_state_dict


class FlowerClient(fl.client.NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        cid: int,
        net: torch.nn.Module,
        train_loader: DataLoader,
        device: str,
        num_epochs: int,
    ):  # pylint: disable=too-many-arguments
        self.cid = cid
        self.net = net
        self.train_loader = train_loader
        self.device = torch.device(device)
        self.num_epochs = num_epochs

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Apply parameters to model state dict."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Train the network on the training set."""
        self.set_parameters(parameters)

        train(
            self.net,
            self.train_loader,
            self.device,
            epochs=self.num_epochs,
            hyperparams=config,
            epoch=int(config["curr_round"]),
        )

        return (
            self.get_parameters({}),
            len(self.train_loader),
            {},
        )


# pylint: disable=too-many-instance-attributes
class PFlowerClient(fl.client.NumPyClient):
    """Personalized Flower Client."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        cid: int,
        net: torch.nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str,
        num_epochs: int,
        state_path: str,
        algorithm: str,
    ):
        self.cid = cid
        self.net = net
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device(device)
        self.num_epochs = num_epochs
        self.state_path = state_path
        self.algorithm = algorithm
        self.private_server_param: Dict[str, torch.Tensor] = {}

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""
        model_dict = self.net.state_dict().copy()
        # overwrite the server private parameters
        for k in self.private_server_param.keys():
            model_dict[k] = self.private_server_param[k]
        return [val.cpu().numpy() for _, val in model_dict.items()]

    def set_parameters(self, parameters: NDArrays, evaluate: bool) -> None:
        """Apply parameters to model state dict."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        server_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        self.private_server_param = {
            k: server_dict[k]
            for k in get_keys_state_dict(
                model=self.net, algorithm=self.algorithm, mode="local"
            )
        }

        if evaluate:
            client_dict = self.net.state_dict().copy()
        else:
            client_dict = copy.deepcopy(server_dict)

        if os.path.isfile(self.state_path):
            with open(self.state_path, "rb") as f:
                client_dict = torch.load(f)

        for k in get_keys_state_dict(
            model=self.net, algorithm=self.algorithm, mode="global"
        ):
            client_dict[k] = server_dict[k]

        self.net.load_state_dict(client_dict, strict=False)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Train the network on the training set."""
        self.set_parameters(parameters, evaluate=False)

        train(
            self.net,
            self.train_loader,
            self.device,
            epochs=self.num_epochs,
            hyperparams=config,
            epoch=int(config["curr_round"]),
        )
        if self.state_path is not None:
            with open(self.state_path, "wb") as f:
                torch.save(self.net.state_dict(), f)

        return (
            self.get_parameters({}),
            len(self.train_loader),
            {},
        )

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Evaluate the network on the test set."""
        self.set_parameters(parameters, evaluate=True)
        self.net.to(self.device)
        loss, accuracy = test(self.net, self.test_loader, device=self.device)
        return loss, len(self.test_loader), {"accuracy": accuracy}


# pylint: disable=too-many-arguments
def gen_client_fn(
    train_loaders: List[DataLoader],
    model: DictConfig,
    num_epochs: int,
    args: Dict,
    test_loader: Optional[List[DataLoader]] = None,
    state_path: Optional[str] = None,
) -> Callable[[str], fl.client.Client]:
    """Return a function which creates a new FlowerClient for a given cid."""

    def client_fn(cid: str) -> fl.client.Client:
        """Create a new FlowerClient for a given cid."""
        cid_ = int(cid)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if args["algorithm"].lower() == "pfedpara" or args["algorithm"] == "fedper":
            cl_path = f"{state_path}/client_{cid_}.pth"
            return PFlowerClient(
                cid=cid_,
                net=instantiate(model).to(device),
                train_loader=train_loaders[cid_],
                test_loader=copy.deepcopy(test_loader),
                num_epochs=num_epochs,
                state_path=cl_path,
                algorithm=args["algorithm"].lower(),
                device=device,
            ).to_client()
        return FlowerClient(
            cid=cid_,
            net=instantiate(model).to(device),
            train_loader=train_loaders[cid_],
            num_epochs=num_epochs,
            device=device,
        ).to_client()

    return client_fn
