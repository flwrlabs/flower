"""cgsv: A Flower Baseline."""
"""Cosine Gradient Shapley Value (CGSV) Strategy implementation for Flower.

This module implements a federated learning strategy that uses cosine similarity between
client gradients to compute importance coefficients and perform gradient sparsification
for personalized federated learning.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import math  # <-- Add this import at the top with your other imports

import numpy as np

from flwr.common import FitRes, NDArrays, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy



class CGSVStrategy(Strategy):
    """Cosine Gradient Shapley Value (CGSV) Strategy."""

    def __init__(
        self,
        alpha: float = 0.95,  # Moving average weight for importance coefficients
        beta: float = 1.5,  # Altruism parameter for sparsification
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[Callable] = None,
        on_fit_config_fn: Optional[Callable] = None,
        on_evaluate_config_fn: Optional[Callable] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[NDArrays] = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters

        # State variables
        self.importance_coefficients: Dict[str, float] = (
            {}
        )  # Tracks client contributions
        self.global_model_weights: Optional[NDArrays] = (
            initial_parameters  # Global model state
        )
        self.sparsified_gradients: Dict[str, NDArrays] = (
            {}
        )  # Stores sparsified gradients per client

    def num_fit_clients(self, num_available: int) -> Tuple[int, int]:
        """Determine the number of clients to sample for training.
        
        Uses the fraction_fit and min_fit_clients to compute the sample size.
        """
        sample_size = int(math.ceil(num_available * self.fraction_fit))
        sample_size = max(sample_size, self.min_fit_clients)
        sample_size = min(sample_size, num_available)
        return sample_size, self.min_fit_clients

    def num_evaluate_clients(self, num_available: int) -> Tuple[int, int]:
        """Determine the number of clients to sample for evaluation.
        
        Uses the fraction_evaluate and min_evaluate_clients to compute the sample size.
        """
        sample_size = int(math.ceil(num_available * self.fraction_evaluate))
        sample_size = max(sample_size, self.min_evaluate_clients)
        sample_size = min(sample_size, num_available)
        return sample_size, self.min_evaluate_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[NDArrays]:
        """Initialize global model parameters."""
        return self.initial_parameters

    def configure_fit(
        self, server_round: int, parameters: NDArrays, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, Dict[str, Scalar]]]:
        """Configure the next round of training."""
        config = {"local_epochs": 1, "batch_size": 32}  # Default config
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Prepare personalized parameters using sparsified gradients
        client_configs = []
        for client in clients:
            cid = client.cid
            if cid in self.sparsified_gradients:
                # Apply sparsified gradient to global model
                sparsified_grad = self.sparsified_gradients[cid]
                personalized_params = [
                    w + g for w, g in zip(self.global_model_weights, sparsified_grad)
                ]
                client_configs.append(
                    (client, {"parameters": personalized_params, "config": config})
                )
            else:
                # First round or new client, use global parameters
                client_configs.append(
                    (client, {"parameters": parameters, "config": config})
                )

        return client_configs

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[NDArrays], Dict[str, Scalar]]:
        """Aggregate fit results using CGSV mechanism."""
        if not results:
            return None, {}

        # Collect normalized gradients from clients
        client_gradients = {}
        for client, fit_res in results:
            cid = client.cid
            gradients = fit_res.parameters  # Normalized gradients from client
            client_gradients[cid] = (gradients, fit_res.num_examples)

        # 1. Aggregate gradients using importance coefficients
        aggregated_grad = self._aggregate_gradients(client_gradients)

        # 2. Compute CGSV (cosine similarity) for each client
        cgsv_scores = {}
        for cid, (grad, _) in client_gradients.items():
            cgsv_scores[cid] = self._cosine_similarity(grad, aggregated_grad)

        # 3. Update importance coefficients
        for cid in cgsv_scores:
            prev_r = self.importance_coefficients.get(cid, 1.0)  # Initialize to 1.0
            new_r = self.alpha * prev_r + (1 - self.alpha) * cgsv_scores[cid]
            self.importance_coefficients[cid] = new_r

        # Normalize importance coefficients
        total_r = sum(self.importance_coefficients.values())
        for cid in self.importance_coefficients:
            self.importance_coefficients[cid] /= total_r

        # 4. Sparsify gradients for each client
        self.sparsified_gradients = {}
        max_tanh = max(
            np.tanh(self.beta * r) for r in self.importance_coefficients.values()
        )
        D = sum([np.prod(g.shape) for g in aggregated_grad])  # Total dimensions

        for cid in self.importance_coefficients:
            r_i = self.importance_coefficients[cid]
            q_i = int(D * np.tanh(self.beta * r_i) / max_tanh)
            sparsified_grad = self._sparsify_gradient(aggregated_grad, q_i)
            self.sparsified_gradients[cid] = sparsified_grad

        # 5. Update global model
        if self.global_model_weights is None:
            self.global_model_weights = aggregated_grad
        else:
            self.global_model_weights = [
                w + g for w, g in zip(self.global_model_weights, aggregated_grad)
            ]

        return self.global_model_weights, {}

    def _aggregate_gradients(
        self, client_gradients: Dict[str, Tuple[NDArrays, int]]
    ) -> NDArrays:
        """Aggregate gradients using importance coefficients."""
        aggregated = [
            np.zeros_like(g) for g in next(iter(client_gradients.values()))[0]
        ]
        for cid, (grad, _) in client_gradients.items():
            weight = self.importance_coefficients.get(cid, 1.0 / len(client_gradients))
            for i in range(len(aggregated)):
                aggregated[i] += grad[i] * weight
        return aggregated

    def _cosine_similarity(self, grad_a: NDArrays, grad_b: NDArrays) -> float:
        """Compute cosine similarity between two gradients."""
        flat_a = np.concatenate([g.flatten() for g in grad_a])
        flat_b = np.concatenate([g.flatten() for g in grad_b])
        dot = np.dot(flat_a, flat_b)
        norm_a = np.linalg.norm(flat_a)
        norm_b = np.linalg.norm(flat_b)
        return dot / (norm_a * norm_b + 1e-8)

    def _sparsify_gradient(self, gradient: NDArrays, q_i: int) -> NDArrays:
        """Sparsify gradient by keeping top-q_i elements by magnitude."""
        if q_i <= 0:
            return [np.zeros_like(g) for g in gradient]

        # Flatten and find threshold
        flat_grad = np.concatenate([g.flatten() for g in gradient])
        abs_grad = np.abs(flat_grad)
        threshold = np.sort(abs_grad)[-q_i] if q_i < len(flat_grad) else 0

        # Create mask
        mask = (abs_grad >= threshold).astype(np.float32)
        sparsified_flat = flat_grad * mask

        # Reshape back to original structure
        sparsified = []
        pointer = 0
        for g in gradient:
            size = np.prod(g.shape)
            sparsified.append(
                sparsified_flat[pointer : pointer + size].reshape(g.shape)
            )
            pointer += size
        return sparsified

    def configure_evaluate(
        self, server_round: int, parameters: NDArrays, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, Dict[str, Scalar]]]:
        """Configure clients for evaluation."""
        if self.on_evaluate_config_fn is not None:
            eval_config = self.on_evaluate_config_fn(server_round)
        else:
            eval_config = {}

        # Sample clients
        sample_size, min_num_clients = self.num_evaluate_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, eval_config) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, Dict[str, Scalar]]],
        failures: List[Union[Tuple[ClientProxy, Dict[str, Scalar]], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        if not results:
            return None, {}

        # Aggregate loss and metrics
        loss_aggregated = sum(r.metrics["loss"] * r.num_examples for _, r in results) / sum(
            r.num_examples for _, r in results
        )

        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: NDArrays
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            return None

        eval_res = self.evaluate_fn(server_round, parameters, {})
        if eval_res is None:
            return None

        loss, metrics = eval_res
        return loss, metrics