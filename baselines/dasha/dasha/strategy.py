"""Defines Flower Strategies."""

import time
from logging import INFO, WARNING
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.server.strategy.strategy import Strategy

from dasha.client import CompressionClient, DashaClient, MarinaClient
from dasha.compressors import IdentityUnbiasedCompressor, decompress, estimate_size


class _CompressionAggregator(Strategy):
    _EMPTY_CONFIG: Dict[str, Scalar] = {}
    _SKIPPED = "skipped"
    ACCURACY = "accuracy"
    SQUARED_GRADIENT_NORM = "squared_gradient_norm"
    RECEIVED_BYTES = "received_bytes"

    def __init__(self, step_size, num_clients):
        self._step_size = step_size
        self._parameters = None
        self._gradient_estimator = None
        self._num_clients = num_clients
        self._total_received_bytes_per_client_during_fit = 0

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        return None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        raise NotImplementedError()

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        evel_ins = EvaluateIns(parameters, self._EMPTY_CONFIG)
        return [(client, evel_ins) for client in client_manager.all().values()]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Implement the server's logic from Algorithm 1 in the DASHA paper.

        (almost the same logic is in the MARINA paper).
        """
        assert len(failures) == 0, failures
        if len(results) != self._num_clients:
            log(WARNING, "not all clients have sent results. Waiting and repeating...")
            time.sleep(1.0)
            return ndarrays_to_parameters([self._parameters]), {}
        parsed_results = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]
        expect_compressor = (
            IdentityUnbiasedCompressor.name()
            if self._gradient_estimator is None
            else None
        )
        estimated_sizes = [
            estimate_size(compressed_params) for compressed_params in parsed_results
        ]
        max_estimated_size = int(np.max(estimated_sizes))
        self._total_received_bytes_per_client_during_fit += max_estimated_size
        parsed_results = [
            decompress(compressed_params, assert_compressor=expect_compressor)
            for compressed_params in parsed_results
        ]
        aggregated_vector = np.add.reduce(parsed_results) / len(parsed_results)
        if self._gradient_estimator is None:
            self._gradient_estimator = aggregated_vector
        else:
            self._gradient_estimator += aggregated_vector
        self._parameters -= self._step_size * self._gradient_estimator
        return ndarrays_to_parameters([self._parameters]), {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Calculate metrics."""
        assert len(failures) == 0
        loss_aggregated = weighted_loss_avg(
            [(1, evaluate_res.loss) for _, evaluate_res in results]
        )
        log(INFO, "Round: %i", server_round)
        log(INFO, "Aggregated loss: %f", loss_aggregated)
        metrics = {
            self.RECEIVED_BYTES: self._total_received_bytes_per_client_during_fit
        }
        if CompressionClient.GRADIENT in results[0][1].metrics:
            gradients: List[bytes] = []
            for _, evaluate_res in results:
                gradient_unparsed = evaluate_res.metrics[CompressionClient.GRADIENT]
                assert isinstance(gradient_unparsed, bytes)
                gradients.append(gradient_unparsed)
            gradients_parsed = [
                np.frombuffer(gradient, dtype=np.float32) for gradient in gradients
            ]
            gradient = np.add.reduce(gradients_parsed) / len(gradients_parsed)
            norm_square = float(np.linalg.norm(gradient) ** 2)
            metrics[self.SQUARED_GRADIENT_NORM] = norm_square
            log(INFO, "Squared gradient norm: %f", norm_square)
        if CompressionClient.ACCURACY in results[0][1].metrics:
            accuracies: List[Tuple[int, float]] = []
            for _, evaluate_res in results:
                accuracy = evaluate_res.metrics[CompressionClient.ACCURACY]
                assert isinstance(accuracy, float)
                accuracies.append((1, accuracy))
            accuracy_aggregated = weighted_loss_avg(accuracies)
            metrics[CompressionClient.ACCURACY] = accuracy_aggregated
            log(INFO, "Aggregated accuracy: %f", accuracy_aggregated)
        return loss_aggregated, metrics

    def evaluate(  # pylint: disable=useless-return
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Initialize the parameters on the server using parameters from a node."""
        if server_round == 0:
            ndarrays = parameters_to_ndarrays(parameters)
            assert len(ndarrays) == 1
            self._parameters = ndarrays[0]
        return None


class DashaAggregator(_CompressionAggregator):
    """Standard Flower strategy."""

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Run configure_fit logic."""
        fit_ins = FitIns(
            parameters,
            {DashaClient.SEND_FULL_GRADIENT: self._gradient_estimator is None},
        )
        return [(client, fit_ins) for client in client_manager.all().values()]


class MarinaAggregator(_CompressionAggregator):
    """Standard Flower strategy."""

    def __init__(self, *args, seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._generator = np.random.default_rng(seed)
        self._size_of_compressed_vectors = None
        self._prob = None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """MARINA asks the nodes to send noncompressed vectors some probability."""
        if self._gradient_estimator is not None:
            prob = self._get_prob()
            if self._bernoulli_sample(self._generator, prob):
                self._gradient_estimator = None
        fit_ins = FitIns(
            parameters,
            {MarinaClient.SEND_FULL_GRADIENT: self._gradient_estimator is None},
        )
        return [(client, fit_ins) for client in client_manager.all().values()]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Run aggregate_fit logic."""
        loss_aggregated, metrics = super().aggregate_fit(
            server_round, results, failures
        )
        size_of_compressed_vectors = [
            fit_res.metrics[CompressionClient.SIZE_OF_COMPRESSED_VECTORS]
            for _, fit_res in results
        ]
        assert np.all(
            np.equal(size_of_compressed_vectors, size_of_compressed_vectors[0])
        )
        self._size_of_compressed_vectors = size_of_compressed_vectors[0]
        return loss_aggregated, metrics

    def _get_prob(self):
        """Find the probability of sending noncompressed vectors."""
        if self._prob is not None:
            return self._prob
        self._prob = self._size_of_compressed_vectors / len(self._parameters)
        return self._prob

    def _bernoulli_sample(self, random_generator, prob):
        if prob == 0.0:
            return False
        return random_generator.random() < prob
