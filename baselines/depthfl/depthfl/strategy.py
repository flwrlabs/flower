"""Strategy for DepthFL."""

import os
import pickle
from logging import WARNING
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from flwr.common import (
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.common.typing import FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from omegaconf import DictConfig


class FedDyn(FedAvg):
    """Applying dynamic regularization in FedDyn paper."""

    def __init__(self, cfg: DictConfig, net: nn.Module, *args, **kwargs):
        self.cfg = cfg
        self.h_variate = [np.zeros(v.shape) for (k, v) in net.state_dict().items()]

        # tagging real weights / biases
        self.is_weight = []
        for k in net.state_dict().keys():
            if "weight" not in k and "bias" not in k:
                self.is_weight.append(False)
            else:
                self.is_weight.append(True)

        # prev_grads file for each client
        prev_grads = [
            {k: torch.zeros(v.numel()) for (k, v) in net.named_parameters()}
        ] * cfg.num_clients

        if not os.path.exists("prev_grads"):
            os.makedirs("prev_grads")

        for idx in range(cfg.num_clients):
            with open(f"prev_grads/client_{idx}", "wb") as prev_grads_file:
                pickle.dump(prev_grads[idx], prev_grads_file)

        super().__init__(*args, **kwargs)


def aggregate_fit_depthfl(
    strategy,
    server_round: int,
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    origin: NDArrays,
) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
    """Aggregate fit results using weighted average."""
    if not results:
        return None, {}
    # Do not aggregate if there are failures and failures are not accepted
    if not strategy.accept_failures and failures:
        return None, {}

    # Convert results
    weights_results = [
        (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
        for _, fit_res in results
    ]
    parameters_aggregated = ndarrays_to_parameters(
        aggregate(
            weights_results,
            origin,
            strategy.h_variate,
            strategy.is_weight,
            strategy.cfg,
        )
    )

    # Aggregate custom metrics if aggregation fn was provided
    metrics_aggregated = {}
    if strategy.fit_metrics_aggregation_fn:
        fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
        metrics_aggregated = strategy.fit_metrics_aggregation_fn(fit_metrics)
    elif server_round == 1:  # Only log this warning once
        log(WARNING, "No fit_metrics_aggregation_fn provided")

    return parameters_aggregated, metrics_aggregated


def aggregate(
    results: List[Tuple[NDArrays, int]],
    origin: NDArrays,
    h_list: List,
    is_weight: List,
    cfg: DictConfig,
) -> NDArrays:
    """Aggregate model parameters with different depths."""
    param_count = [0] * len(origin)
    weights_sum = [np.zeros(v.shape) for v in origin]

    # summation & counting of parameters
    for parameters, _ in results:
        for i, layer in enumerate(parameters):
            weights_sum[i] += layer
            param_count[i] += 1

    # update parameters
    for i, weight in enumerate(weights_sum):
        if param_count[i] > 0:
            weight = weight / param_count[i]
            # print(np.isscalar(weight))

            # update h variable for FedDyn
            h_list[i] = (
                h_list[i]
                - cfg.fit_config.alpha
                * param_count[i]
                * (weight - origin[i])
                / cfg.num_clients
            )

            # applying h only for weights / biases
            if is_weight[i] and cfg.fit_config.feddyn:
                weights_sum[i] = weight - h_list[i] / cfg.fit_config.alpha
            else:
                weights_sum[i] = weight

        else:
            weights_sum[i] = origin[i]

    return weights_sum
