"""Defines a custom Server."""

import timeit
from logging import INFO
from typing import Optional

from flwr.common.logger import log
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.history import History
from flwr.server.server import Server
from flwr.server.strategy import Strategy


class MyServer(Server):
    """Custom Flower server.

    This customization of the server has the only scope to allow to start the training
    from a starting round different from 1. This is useful when you want to stop and
    restart the training (saving and loading the state obviously).
    """

    def __init__(
        self,
        *,
        client_manager: Optional[ClientManager] = None,
        strategy: Optional[Strategy] = None,
        starting_round: int = 1,
    ) -> None:
        if client_manager is None:
            client_manager = SimpleClientManager()
        super().__init__(client_manager=client_manager, strategy=strategy)
        print(self.client_manager)
        self.starting_round = starting_round

    # overwriting
    def fit(  # pylint: disable=too-many-locals
        self, num_rounds: int, timeout: Optional[float]
    ) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        if self.starting_round == 1:
            log(INFO, "Evaluating initial parameters")
            res = self.strategy.evaluate(
                self.starting_round - 1, parameters=self.parameters
            )
            if res is not None:
                log(
                    INFO,
                    "initial parameters (loss, other metrics): %s, %s",
                    res[0],
                    res[1],
                )
                history.add_loss_centralized(
                    server_round=self.starting_round - 1, loss=res[0]
                )
                history.add_metrics_centralized(
                    server_round=self.starting_round - 1, metrics=res[1]
                )

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        # changed this
        # for current_round in range(1, num_rounds + 1):
        for current_round in range(
            self.starting_round, self.starting_round + num_rounds
        ):
            # Train model and replace previous global model
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history
