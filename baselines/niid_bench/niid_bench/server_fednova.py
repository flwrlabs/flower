"""Server class for FedNova."""

from logging import DEBUG, INFO

from flwr.common import parameters_to_ndarrays
from flwr.common.logger import log
from flwr.common.typing import Dict, Optional, Parameters, Scalar, Tuple
from flwr.server.client_manager import ClientManager
from flwr.server.server import FitResultsAndFailures, Server, fit_clients

from niid_bench.strategy import FedNovaStrategy


class FedNovaServer(Server):
    """Implement server for FedNova."""

    def __init__(
        self,
        *,
        client_manager: ClientManager,
        strategy: Optional[FedNovaStrategy] = None,
    ) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.strategy: FedNovaStrategy = (
            strategy if strategy is not None else FedNovaStrategy()
        )

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        params_np = parameters_to_ndarrays(self.parameters)
        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit_custom(
            server_round, params_np, results, failures
        )

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)
