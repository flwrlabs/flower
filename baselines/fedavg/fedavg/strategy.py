"""Optionally define a custom strategy.

Needed only when the strategy is not yet implemented in Flower or because you want to
extend or modify the functionality of an existing strategy.
"""
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import torch
from flwr.common import Metrics
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import call, instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Define aggregation function for weighted average during evaluation.

    Parameters
    ----------
    metrics : List[Tuple[int, Metrics]]
        The list of metrics to aggregate.

    Returns
    -------
    Metrics
        The weighted average metric.
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * float(m["accuracy"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": int(sum(accuracies)) / int(sum(examples))}


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
        net = instantiate(client_cfg.model_cfg)
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        loss, accuracy = call(client_cfg.evaluate_fn, net, testloader, device=device)
        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate
