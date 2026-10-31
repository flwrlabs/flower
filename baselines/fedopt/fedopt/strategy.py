"""Optionally define a custom strategy.

Needed only when the strategy is not yet implemented in Flower or because you want to
extend or modify the functionality of an existing strategy.
"""

from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple

import torch
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import call
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    client_cfg: DictConfig,
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
    client_cfg: DictConfig
        Config parameterizing the model to use and how to evaluate it.

    Returns
    -------
    Callable[[int, NDArrays, Dict[str, Scalar]],
            Optional[Tuple[float, Dict[str, Scalar]]]]
        The centralized evaluation function.
    """

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # pylint: disable=unused-argument
        """Use the entire MNIST test set for evaluation."""
        # determine device
        net = call(client_cfg.model_cfg)
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        loss, accuracy = call(client_cfg.evaluate_fn, net, testloader, device=device)
        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate
