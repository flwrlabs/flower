"""Optionally define a custom strategy.

Needed only when the strategy is not yet implemented in Flower or because you want to
extend or modify the functionality of an existing strategy.
"""

from collections import OrderedDict
from logging import WARNING
from typing import Dict, List, Optional, Tuple, Union

import torch
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Metrics,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from fedmeta.models import FemnistNetwork, StackedLSTM
from fedmeta.utils import update_ema


# pylint: disable=too-many-arguments
def fedmeta_update_meta_sgd(
    net: torch.nn.Module,
    alpha: torch.nn.ParameterList,
    beta: float,
    weights_results: NDArrays,
    gradients_aggregated: NDArrays,
    weight_decay: float,
) -> Tuple[NDArrays, torch.nn.ParameterList]:
    """Update model parameters for FedMeta(Meta-SGD).

    Parameters
    ----------
    net : torch.nn.Module
        The list of metrics to aggregate.
    alpha : torch.nn.ParameterList
        alpha is the learning rate. it is updated with parameters in FedMeta (Meta-SGD).
    beta : float
        beta is the learning rate for updating parameters and alpha on the server.
    weights_results : List[Tuple[NDArrays, int]]
        These are the global model parameters for the current round.
    gradients_aggregated : List[Tuple[NDArrays, int]]
        Weighted average of the gradient in the current round.
    WD : float
        The weight decay for Adam optimizer

    Returns
    -------
    weights_prime : List[Tuple[NDArrays, int]]
        These are updated parameters.
    alpha : torch.nn.ParameterLis
        These are updated alpha.
    """
    params_dict = zip(net.state_dict().keys(), weights_results)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    optimizer = torch.optim.Adam(
        list(net.parameters()) + list(alpha), lr=beta, weight_decay=weight_decay
    )
    for params, grad_ins, alphas in zip(net.parameters(), gradients_aggregated, alpha):
        params.grad = torch.tensor(grad_ins).to(params.dtype)
        alphas.grad = torch.tensor(grad_ins).to(params.dtype)
    optimizer.step()
    optimizer.zero_grad()
    weights_prime = [val.cpu().numpy() for _, val in net.state_dict().items()]

    return weights_prime, alpha


def fedmeta_update_maml(
    net: torch.nn.Module,
    beta: float,
    weights_results: NDArrays,
    gradients_aggregated: NDArrays,
    weight_decay: float,
) -> NDArrays:
    """Update model parameters for FedMeta(Meta-SGD).

    Parameters
    ----------
    net : torch.nn.Module
        The list of metrics to aggregate.
    beta : float
        beta is the learning rate for updating parameters on the server.
    weights_results : List[Tuple[NDArrays, int]]
        These are the global model parameters for the current round.
    gradients_aggregated : List[Tuple[NDArrays, int]]
        Weighted average of the gradient in the current round.
    WD : float
        The weight decay for Adam optimizer

    Returns
    -------
    weights_prime : List[Tuple[NDArrays, int]]
        These are updated parameters.
    """
    params_dict = zip(net.state_dict().keys(), weights_results)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    optimizer = torch.optim.Adam(
        list(net.parameters()), lr=beta, weight_decay=weight_decay
    )
    for params, grad_ins in zip(net.parameters(), gradients_aggregated):
        params.grad = torch.tensor(grad_ins).to(params.dtype)
    optimizer.step()
    optimizer.zero_grad()
    weights_prime = [val.cpu().numpy() for _, val in net.state_dict().items()]

    return weights_prime


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate using a weighted average during evaluation.

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
    correct = [num_examples * float(m["correct"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": float(sum(correct)) / float(sum(examples))}


class FedMeta(FedAvg):
    """FedMeta averages the gradient and server parameter update through it."""

    def __init__(self, alpha, beta, data, algo, **kwargs):
        super().__init__(**kwargs)
        self.algo = algo
        self.data = data
        self.beta = beta
        self.ema_loss = None
        self.ema_acc = None

        if self.data == "femnist":
            self.net = FemnistNetwork()
        elif self.data == "shakespeare":
            self.net = StackedLSTM()

        self.alpha = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.full_like(p, alpha))
                for p in self.net.parameters()
            ]
        )

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {"alpha": self.alpha, "algo": self.algo, "data": self.data}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(  # type: ignore
            num_clients=sample_size,
            min_num_clients=min_num_clients,
            server_round=server_round,
            step="fit",
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {"alpha": self.alpha, "algo": self.algo, "data": self.data}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(  # type: ignore
            num_clients=sample_size,
            min_num_clients=min_num_clients,
            server_round=server_round,
            step="evaluate",
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results: List[Tuple[NDArrays, int]] = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        parameters_aggregated = aggregate(weights_results)
        if self.data == "femnist":
            weight_decay_ = 0.001
        else:
            weight_decay_ = 0.0001

        # Gradient Average and Update Parameter for FedMeta(MAML)
        if self.algo == "fedmeta_maml":
            grads_results: List[Tuple[NDArrays, int]] = [
                (fit_res.metrics["grads"], fit_res.num_examples)  # type: ignore
                for _, fit_res in results
            ]
            gradients_aggregated = aggregate(grads_results)
            weights_prime = fedmeta_update_maml(
                self.net,
                self.beta,
                weights_results[0][0],
                gradients_aggregated,
                weight_decay_,
            )
            parameters_aggregated = weights_prime

        # Gradient Average and Update Parameter for FedMeta(Meta-SGD)
        elif self.algo == "fedmeta_meta_sgd":
            grads_results: List[Tuple[NDArrays, int]] = [  # type: ignore
                (fit_res.metrics["grads"], fit_res.num_examples)
                for _, fit_res in results
            ]
            gradients_aggregated = aggregate(grads_results)
            weights_prime, update_alpha = fedmeta_update_meta_sgd(
                self.net,
                self.alpha,
                self.beta,
                weights_results[0][0],
                gradients_aggregated,
                weight_decay_,
            )
            self.alpha = update_alpha
            parameters_aggregated = weights_prime

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return ndarrays_to_parameters(parameters_aggregated), metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        if self.data == "femnist":
            smoothing_weight = 0.95
        else:
            smoothing_weight = 0.7
        self.ema_loss = update_ema(self.ema_loss, loss_aggregated, smoothing_weight)
        loss_aggregated = self.ema_loss

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
            self.ema_acc = update_ema(
                self.ema_acc,
                round(float(metrics_aggregated["accuracy"] * 100), 3),
                smoothing_weight,
            )
            metrics_aggregated["accuracy"] = self.ema_acc

        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated
