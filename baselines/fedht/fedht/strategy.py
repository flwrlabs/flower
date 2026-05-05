"""FedHT and FedIterHT strategies for federated sparse learning.

Paper: Federated Nonconvex Sparse Learning (Tong et al., 2021)
https://arxiv.org/abs/2101.00052
"""

from __future__ import annotations

import numpy as np
from flwr.common import FitIns, NDArrays, Parameters
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


def hard_threshold(arrays: NDArrays, tau: int) -> NDArrays:
    """Keep the tau largest-magnitude entries globally; zero out the rest."""
    sizes = [a.size for a in arrays]
    flat = np.concatenate([a.flatten() for a in arrays])

    if tau >= flat.size:
        return arrays

    abs_flat = np.abs(flat)
    cutoff = np.partition(abs_flat, flat.size - tau)[flat.size - tau]
    mask = abs_flat >= cutoff

    # Resolve ties: keep exactly tau entries when multiple values equal the cutoff.
    nonzero_indices = np.where(mask)[0]
    if nonzero_indices.size > tau:
        excess = nonzero_indices.size - tau
        mask[nonzero_indices[:excess]] = False

    thresholded = flat * mask

    result: NDArrays = []
    start = 0
    for original, size in zip(arrays, sizes):
        result.append(thresholded[start : start + size].reshape(original.shape))
        start += size
    return result


class FedHT(FedAvg):
    """Fed-HT (Algorithm 1 of the paper).

    Clients run plain SGD locally. The server applies H_tau after weighted aggregation.
    """

    def __init__(self, tau: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tau = tau

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict]:
        """Aggregate client updates then apply H_tau."""
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            ndarrays = hard_threshold(ndarrays, self.tau)
            aggregated_parameters = ndarrays_to_parameters(ndarrays)

        return aggregated_parameters, metrics


class FedIterHT(FedHT):
    """FedIter-HT: clients also apply H_tau after each local SGD step (Algorithm 2)."""

    def __init__(self, tau: int, **kwargs) -> None:
        super().__init__(tau=tau, **kwargs)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Extend fit config to signal clients to apply local thresholding."""
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)

        updated = []
        for client, fit_ins in fit_ins_list:
            config = dict(fit_ins.config)
            config["use_local_ht"] = True
            config["tau"] = self.tau
            updated.append((client, FitIns(fit_ins.parameters, config)))

        return updated
