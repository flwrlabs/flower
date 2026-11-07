"""Client for FedExp."""

import copy
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import flwr as fl
import torch
from flwr.common import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


class FlowerClient(fl.client.NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        cid: int,
        net: torch.nn.Module,
        train_loader: DataLoader,
        num_epochs: int,
        data_ratio,
    ):  # pylint: disable=too-many-arguments
        print(f"Initializing Client {cid}")
        self.cid = cid
        self.train_loader = train_loader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = net.to(self.device)
        self.num_epochs = num_epochs
        self.data_ratio = data_ratio

    def _set_parameters(self, parameters: NDArrays) -> None:
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Train the network on the training set."""
        self._set_parameters(parameters)
        print(f"Client {self.cid} Training...")
        prev_net = copy.deepcopy(self.net)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=config["eta_l"],
            momentum=0,
            weight_decay=config["weight_decay"],
        )

        self.net.train()
        counter = 0
        while True:
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                if config["use_data_augmentation"]:
                    transform_train = transforms.Compose(
                        [
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                        ]
                    )
                    images = transform_train(images)
                self.net.zero_grad()
                log_probs = self.net(images)
                loss = criterion(log_probs, labels)
                loss.backward()

                if config["use_gradient_clipping"]:
                    torch.nn.utils.clip_grad_norm_(
                        parameters=self.net.parameters(), max_norm=config["max_norm"]
                    )
                optimizer.step()
                counter += 1
                if counter >= self.num_epochs:
                    break
            if counter >= self.num_epochs:
                break

        with torch.no_grad():
            vec_curr = parameters_to_vector(self.net.parameters())
            vec_prev = parameters_to_vector(prev_net.parameters())
            params_delta_vec = vec_curr - vec_prev
            grad = params_delta_vec

        return (
            [],
            len(self.train_loader),
            {
                "data_ratio": self.data_ratio,
                "grad": grad.to("cpu"),
            },
        )


def gen_client_fn(
    train_loaders: List[DataLoader],
    model: DictConfig,
    num_epochs: int,
    args: Dict,
) -> Callable[[str], FlowerClient]:
    """Return a function which creates a new FlowerClient for a given cid."""

    def client_fn(cid: str) -> FlowerClient:
        """Create a new FlowerClient for a given cid."""

        return FlowerClient(
            cid=int(cid),
            net=instantiate(model),
            train_loader=train_loaders[int(cid)],
            num_epochs=num_epochs,
            data_ratio=args["data_ratio"][int(cid)],
        )

    return client_fn
