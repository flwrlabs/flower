import struct
from ast import Bytes
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import EvaluateRes, FitRes, NDArrays, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg


class FedAvgCpp(FedAvg):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
            # Callable[[NDArrays], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
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
        )

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
        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]
        aggregated_weights = aggregate(weights_results)
        parameters_results = weights_to_parameters(aggregated_weights)

        return parameters_results, {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        loss_aggregated = weighted_loss_avg(
            [
                (
                    evaluate_res.num_examples,
                    evaluate_res.loss,
                )
                for _, evaluate_res in results
            ]
        )

        num_total_evaluation_examples = sum(
            [evaluate_res.num_examples for _, evaluate_res in results]
        )
        weighted_metric = [
            evaluate_res.num_examples * evaluate_res.metrics["loss"]
            for _, evaluate_res in results
        ]

        metrics_aggregated = {}
        metrics_aggregated["loss"] = (
            sum(weighted_metric) / num_total_evaluation_examples
        )

        return loss_aggregated, metrics_aggregated


def weights_to_parameters(weights) -> Parameters:
    tensors = [ndarray_to_bytes(tensor) for tensor in weights]
    return Parameters(tensors=tensors, tensor_type="cpp_double")


def parameters_to_weights(parameters: Parameters) -> NDArrays:
    """Convert parameters object to NumPy weights."""
    weights = [bytes_to_ndarray(tensor) for tensor in parameters.tensors]
    return weights


def bytes_to_ndarray(tensor_bytes: Bytes) -> np.ndarray:
    list_doubles = []
    for idx in range(0, len(tensor_bytes), 8):
        this_double = struct.unpack("d", tensor_bytes[idx : idx + 8])
        list_doubles.append(this_double[0])
    weights_np = np.asarray(list_doubles)
    return weights_np


def ndarray_to_bytes(a: np.ndarray) -> Bytes:
    doublelist = a.tolist()
    buf = struct.pack("%sd" % len(doublelist), *doublelist)
    return buf
