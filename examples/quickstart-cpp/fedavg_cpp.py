"""Strategy helpers for the C++ quickstart example."""

import struct
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import EvaluateRes, FitRes, NDArrays, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg


class FedAvgCpp(FedAvg):
    """FedAvg strategy that understands the C++ example tensor encoding."""

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
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        return ndarrays_to_parameters(aggregate(weights_results)), {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results]
        )
        total_examples = sum(evaluate_res.num_examples for _, evaluate_res in results)
        weighted_metric = sum(
            evaluate_res.num_examples * float(evaluate_res.metrics["loss"])
            for _, evaluate_res in results
        )
        return loss_aggregated, {"loss": weighted_metric / total_examples}


def ndarrays_to_parameters(weights: NDArrays) -> Parameters:
    return Parameters(
        tensors=[ndarray_to_bytes(tensor) for tensor in weights],
        tensor_type="cpp_double",
    )


def parameters_to_ndarrays(parameters: Parameters) -> NDArrays:
    return [bytes_to_ndarray(tensor) for tensor in parameters.tensors]


def bytes_to_ndarray(tensor_bytes: bytes) -> np.ndarray:
    if len(tensor_bytes) % 8 != 0:
        raise ValueError("C++ tensor byte length must be divisible by 8")
    values = [
        struct.unpack("<d", tensor_bytes[idx : idx + 8])[0]
        for idx in range(0, len(tensor_bytes), 8)
    ]
    return np.asarray(values, dtype=np.float64)


def ndarray_to_bytes(array: np.ndarray) -> bytes:
    values = np.asarray(array, dtype=np.float64).tolist()
    return struct.pack("<%sd" % len(values), *values)
