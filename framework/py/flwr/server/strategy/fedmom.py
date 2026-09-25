# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Federated Momentum (FedMom) [Huo et al., 2020] strategy.

Paper: arxiv.org/abs/2002.02090
"""

from collections.abc import Callable
from logging import WARNING

from flwr.common import (
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from .aggregate import aggregate
from .fedavg import FedAvg


# pylint: disable=line-too-long
class FedMom(FedAvg):
    """Federated Momentum strategy.

    Implementation based on https://arxiv.org/abs/2002.02090

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional. Defaults to None.
    evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional. Defaults to None.
    server_learning_rate : float, optional
        Server-side learning rate used in server-side optimization.
        Defaults to 1.0.
    server_momentum : float, optional
        Server-side momentum factor used for FedMom. Defaults to 0.9.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: (
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                tuple[float, dict[str, Scalar]] | None,
            ]
            | None
        ) = None,
        on_fit_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        on_evaluate_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        accept_failures: bool = True,
        initial_parameters: Parameters | None = None,
        fit_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        server_learning_rate: float = 1.0,
        server_momentum: float = 0.9,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.server_learning_rate = server_learning_rate
        self.server_momentum = server_momentum
        self.v_vector: NDArrays | None = None

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedMom(accept_failures={self.accept_failures})"
        return rep

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        """Initialize global model parameters."""
        return self.initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        self.initial_parameters = parameters
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate fit results using weighted average and server momentum."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        fedavg_result = aggregate(weights_results)

        if self.initial_parameters is None:
            raise ValueError(
                "When using server-side optimization, model needs to be initialized."
            )
        w_t = parameters_to_ndarrays(self.initial_parameters)

        # Compute pseudo-gradient: g_t = w_t - w_{avg, t+1}
        pseudo_gradient: NDArrays = [
            w - w_avg for w, w_avg in zip(w_t, fedavg_result, strict=True)
        ]

        # Update momentum vector: v_{t+1} = w_t - eta * g_t
        v_next: NDArrays = [
            w - self.server_learning_rate * pg
            for w, pg in zip(w_t, pseudo_gradient, strict=True)
        ]

        # Determine previous momentum vector v_t (v_0 = w_0 for first round)
        v_prev: NDArrays = w_t if self.v_vector is None else self.v_vector

        # Update global model parameters: w_{t+1} = v_{t+1} + beta * (v_{t+1} - v_t)
        w_next: NDArrays = [
            vn + self.server_momentum * (vn - vp)
            for vn, vp in zip(v_next, v_prev, strict=True)
        ]

        parameters_aggregated = ndarrays_to_parameters(w_next)

        # Update internal state
        self.v_vector = v_next
        self.initial_parameters = parameters_aggregated

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
