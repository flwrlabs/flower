"""FedExP strategy."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from fedexp.utils import get_parameters


class FedExP(FedAvg):
    """FedExP strategy."""

    def __init__(
        self,
        algorithm: str,
        net_glob: torch.nn.Module,
        epsilon: float,
        decay: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.algorithm = algorithm
        self.net_glob = net_glob
        self.epsilon = epsilon
        self.decay = decay
        self.w_vec_estimate = np.zeros(
            parameters_to_vector(self.net_glob.parameters()).numel()
        )

    def __repr__(self) -> str:
        """Return the name of the strategy."""
        return self.algorithm

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using FedExP."""
        if not self.accept_failures and failures:
            return None, {}
        # Aggregate results
        clients_per_round = len(results)
        self.epsilon *= self.decay ** 2
        p_sum = sum([res.metrics["data_ratio"] for _, res in results])
        d = parameters_to_vector(self.net_glob.parameters()).numel()
        grad_avg = torch.zeros(d)
        grad_norm_sum = 0
        for _, res in results:
            grad_norm_sum += res.metrics["data_ratio"] * torch.linalg.norm(res.metrics["grad"]) ** 2
            grad_avg += res.metrics["data_ratio"] * res.metrics["grad"]
        with torch.no_grad():
            grad_avg /= p_sum
            grad_norm_avg = grad_norm_sum / p_sum
            grad_avg_norm = torch.linalg.norm(grad_avg) ** 2

            if self.algorithm.lower() == "fedexp":
                eta_g = max(
                    1,
                    (
                        0.5
                        * grad_norm_avg
                        / (grad_avg_norm + clients_per_round * self.epsilon).cpu()
                    ).item(),
                )
            elif self.algorithm.lower() == "fedavg":
                eta_g = 1
            else:
                raise NotImplementedError(f"Algorithm {self.algorithm} not implemented.")

            w_vec_prev = self.w_vec_estimate
            self.w_vec_estimate = (
                parameters_to_vector(self.net_glob.parameters()) + eta_g * grad_avg
            )
            w_vec_avg = (
                self.w_vec_estimate
                if server_round == 0
                else (self.w_vec_estimate + w_vec_prev) / 2
            )

            if self.algorithm.lower() == "fedexp":
                vector_to_parameters(w_vec_avg, self.net_glob.parameters())
            elif self.algorithm.lower() == "fedavg":
                vector_to_parameters(self.w_vec_estimate, self.net_glob.parameters())

        return ndarrays_to_parameters(get_parameters(self.net_glob)), {"eta_g": eta_g}
