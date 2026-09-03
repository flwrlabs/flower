"""Defines the client class and support functions for SCAFFOLD."""

import os
from typing import Callable, Dict, List

import flwr as fl
import torch
from flwr.common import Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from niid_bench.models import test, train_scaffold


# pylint: disable=too-many-instance-attributes
class FlowerClientScaffold(fl.client.NumPyClient):
    """Flower client implementing scaffold."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        cid: int,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
        save_dir: str = "",
    ) -> None:
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        # initialize client control variate with 0 and shape of the network parameters
        self.client_cv = []
        for param in self.net.parameters():
            self.client_cv.append(torch.zeros(param.shape))
        # save cv to directory
        if save_dir == "":
            save_dir = "client_cvs"
        self.dir = save_dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Return the current local model parameters."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """Set the local model parameters using given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = {k: torch.from_numpy(v) for k, v in params_dict}
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config: Dict[str, Scalar]):
        """Implement distributed fit function for a given client for SCAFFOLD."""
        # the first half are model parameters and the second are the server_cv
        server_cv = parameters[len(parameters) // 2 :]
        parameters = parameters[: len(parameters) // 2]
        self.set_parameters(parameters)
        self.client_cv = []
        for param in self.net.parameters():
            self.client_cv.append(param.clone().detach())
        # load client control variate
        if os.path.exists(f"{self.dir}/client_cv_{self.cid}.pt"):
            self.client_cv = torch.load(f"{self.dir}/client_cv_{self.cid}.pt")
        # convert the server control variate to a list of tensors
        server_cv = [torch.Tensor(cv) for cv in server_cv]
        train_scaffold(
            self.net,
            self.trainloader,
            self.device,
            self.num_epochs,
            self.learning_rate,
            self.momentum,
            self.weight_decay,
            server_cv,
            self.client_cv,
        )
        x = parameters
        y_i = self.get_parameters(config={})
        c_i_n = []
        server_update_x = []
        server_update_c = []
        # update client control variate c_i_1 = c_i - c + 1/eta*K (x - y_i)
        for c_i_j, c_j, x_j, y_i_j in zip(self.client_cv, server_cv, x, y_i):
            c_i_n.append(
                c_i_j
                - c_j
                + (1.0 / (self.learning_rate * self.num_epochs * len(self.trainloader)))
                * (x_j - y_i_j)
            )
            # y_i - x, c_i_n - c_i for the server
            server_update_x.append((y_i_j - x_j))
            server_update_c.append((c_i_n[-1] - c_i_j).cpu().numpy())
        self.client_cv = c_i_n
        torch.save(self.client_cv, f"{self.dir}/client_cv_{self.cid}.pt")

        combined_updates = server_update_x + server_update_c

        return (
            combined_updates,
            len(self.trainloader.dataset),
            {},
        )

    def evaluate(self, parameters, config: Dict[str, Scalar]):
        """Evaluate using given parameters."""
        self.set_parameters(parameters)
        loss, acc = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(acc)}


# pylint: disable=too-many-arguments
def gen_client_fn(
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    client_cv_dir: str,
    num_epochs: int,
    learning_rate: float,
    model: DictConfig,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
) -> Callable[[str], FlowerClientScaffold]:  # pylint: disable=too-many-arguments
    """Generate the client function that creates the scaffold flower clients.

    Parameters
    ----------
    trainloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset training partition
        belonging to a particular client.
    valloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset validation partition
        belonging to a particular client.
    client_cv_dir : str
        The directory where the client control variates are stored (persistent storage).
    num_epochs : int
        The number of local epochs each client should run the training for before
        sending it to the server.
    learning_rate : float
        The learning rate for the SGD  optimizer of clients.
    momentum : float
        The momentum for SGD optimizer of clients.
    weight_decay : float
        The weight decay for SGD optimizer of clients.

    Returns
    -------
    Callable[[str], FlowerClientScaffold]
        The client function that creates the scaffold flower clients.
    """

    def client_fn(cid: str) -> FlowerClientScaffold:
        """Create a Flower client representing a single organization."""
        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = instantiate(model).to(device)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        return FlowerClientScaffold(
            int(cid),
            net,
            trainloader,
            valloader,
            device,
            num_epochs,
            learning_rate,
            momentum,
            weight_decay,
            save_dir=client_cv_dir,
        )

    return client_fn
