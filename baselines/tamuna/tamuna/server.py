"""Centralized server evaluation function generation."""

from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple

import torch
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from tamuna.models import test


def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    model: DictConfig,
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Generate the function for centralized evaluation.

    Parameters
    ----------
    testloader : DataLoader
        The dataloader to test the model with.
    device : torch.device
        The device to test the model on.
    model: DictConfig
        Architecture of the model being evaluated.

    Returns
    -------
    Callable[[NDArrays], Tuple[float, Dict[str, Scalar]]]
        The centralized evaluation function.
    """

    # pylint: disable=unused-argument
    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Use the entire MNIST test set for evaluation."""
        net = instantiate(model)
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        loss, accuracy = test(net, testloader, device=device)
        return loss, {"accuracy": accuracy}

    return evaluate
