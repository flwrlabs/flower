"""Strategy for HeteroFL."""

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
from hydra.utils import instantiate
from omegaconf import DictConfig


class HeteroFL(FedAvg):
    """Custom FedAvg for HeteroFL."""

    def __init__(self, cfg: DictConfig, net: nn.Module, *args, **kwargs):
        self.cfg = cfg
        self.parameters = [np.zeros(v.shape) for (k, v) in net.state_dict().items()]
        self.param_idx_lst = []

        model = cfg.model
        # store parameter shapes of different width
        for i in range(4):
            model.n_blocks = i + 1
            net_tmp = instantiate(model)
            param_idx = []
            for k in net_tmp.state_dict().keys():
                param_idx.append(
                    [torch.arange(size) for size in net_tmp.state_dict()[k].shape]
                )

            # print(net_tmp.state_dict()['conv1.weight'].shape[0])
            self.param_idx_lst.append(param_idx)

        self.is_weight = []

        # tagging real weights / biases
        for k in net.state_dict().keys():
            if "num" in k:
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

    def aggregate_hetero(
        self, results: List[Tuple[NDArrays, Union[bool, bytes, float, int, str]]]
    ):
        """Aggregate function for HeteroFL."""
        for i, params in enumerate(self.parameters):
            count = np.zeros(params.shape)
            tmp_v = np.zeros(params.shape)
            if self.is_weight[i]:
                for weights, cid in results:
                    if self.cfg.exclusive_learning:
                        cid = self.cfg.model_size * (self.cfg.num_clients // 4) - 1

                    tmp_v[
                        torch.meshgrid(
                            self.param_idx_lst[cid // (self.cfg.num_clients // 4)][i]
                        )
                    ] += weights[i]
                    count[
                        torch.meshgrid(
                            self.param_idx_lst[cid // (self.cfg.num_clients // 4)][i]
                        )
                    ] += 1
                tmp_v[count > 0] = np.divide(tmp_v[count > 0], count[count > 0])
                params[count > 0] = tmp_v[count > 0]

            else:
                for weights, _ in results:
                    tmp_v += weights[i]
                    count += 1
                tmp_v = np.divide(tmp_v, count)
                params = tmp_v


def aggregate_fit_hetero(
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
        (parameters_to_ndarrays(fit_res.parameters), fit_res.metrics["cid"])
        for _, fit_res in results
    ]

    strategy.parameters = origin
    strategy.aggregate_hetero(weights_results)
    parameters_aggregated = ndarrays_to_parameters(strategy.parameters)

    # Aggregate custom metrics if aggregation fn was provided
    metrics_aggregated = {}
    if strategy.fit_metrics_aggregation_fn:
        fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
        metrics_aggregated = strategy.fit_metrics_aggregation_fn(fit_metrics)
    elif server_round == 1:  # Only log this warning once
        log(WARNING, "No fit_metrics_aggregation_fn provided")

    return parameters_aggregated, metrics_aggregated
