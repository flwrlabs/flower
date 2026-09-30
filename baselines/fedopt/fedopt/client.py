"""Simple Flower Client."""

from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import flwr as fl
import torch
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import call
from omegaconf import DictConfig
from torch.utils.data import DataLoader


class FlowerClient(fl.client.NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        net: torch.nn.Module,
        trainloader: DataLoader,
        client_cfg: DictConfig,
    ):
        self.net = net
        self.trainloader = trainloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_fn = client_cfg.train_fn

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
        call(
            self.train_fn,
            net=self.net,
            trainloader=self.trainloader,
            device=self.device,
        )
        return self.get_parameters({}), len(self.trainloader), {}


def gen_client_fn(
    trainloaders: List[DataLoader],
    client_cfg: DictConfig,
) -> Callable[[str], FlowerClient]:
    """Generate the client function that creates the Flower Clients.

    Parameters
    ----------
    trainloaders : List[DataLoader]
        List of train dataloaders to be used by clients.
    client_cfg: DictConfig
        Config to be used to instantiate the model via Hydra's instantiate and
        further parameterise the client's fit() and evaluate() behaviour.

    Returns
    -------
    Tuple[Callable[[str], FlowerClient], DataLoader]
        A tuple containing the client function that creates Flower Clients and
        the DataLoader that will be used for testing.
    """

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""
        # Load model
        net = call(client_cfg.model_cfg)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]

        # Create a  single Flower client representing a single organization
        return FlowerClient(net, trainloader, client_cfg)

    return client_fn
