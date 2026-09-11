"""Global evaluation function."""

from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import torch
from flwr.common import Metrics, NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from fedpara.models import test


def get_on_fit_config(hypearparams: Dict):
    """Generate fit config function."""

    def fit_config_fn(server_round: int):
        hypearparams["curr_round"] = server_round
        return hypearparams

    return fit_config_fn


def gen_evaluate_fn(
    num_clients: int,
    test_loader: DataLoader,
    model: DictConfig,
    device,
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Generate a centralized evaluation function.

    Parameters
    ----------
    model: DictConfig
        The model details to evaluate.
    test_loader : DataLoader
        The dataloader to test the model with.
    device : torch.device
        The device to test the model on.

    Returns
    -------
    Callable[ [int, NDArrays, Dict[str, Scalar]],
            Optional[Tuple[float, Dict[str, Scalar]]] ]
        The centralized evaluation function.
    """

    # pylint: disable=unused-argument
    def evaluate(
        server_round,
        parameters_ndarrays: NDArrays,
        __,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Use the entire CIFAR-10/100 test set for evaluation."""
        net = instantiate(model)
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        loss, accuracy = test(net, test_loader, device=device)
        return loss, {"accuracy": accuracy}

    return evaluate


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Do weighted average of metrics."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [float(num_examples * m["accuracy"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    print(f"accuracies: {sum(accuracies) / sum(examples)}")
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}
