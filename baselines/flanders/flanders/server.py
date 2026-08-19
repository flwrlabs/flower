"""Server with enhanced functionality.

It can be used to simulate an attacker that controls a fraction of the clients and to
save the parameters of each client in its memory.
"""

import timeit
from logging import DEBUG, INFO
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from flwr.common import DisconnectRes, EvaluateRes, FitRes, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.server import Server, fit_clients

from .strategy import Flanders
from .utils import flatten_params, save_params, update_confusion_matrix

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, DisconnectRes]],
    List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]],
]


class EnhancedServer(Server):
    """Server with enhanced functionality."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        num_malicious: int,
        warmup_rounds: int,
        attack_fn: Callable,
        dataset_name: str,
        *args: Any,
        threshold: float = 0.0,
        to_keep: int = 1,
        magnitude: float = 0.0,
        sampling: int = 0,
        history_dir: str = "clients_params",
        omniscent: bool = True,
        **kwargs: Any,
    ) -> None:
        """Create a new EnhancedServer instance.

        Parameters
        ----------
        num_malicious : int
            Number of malicious clients
        warmup_rounds : int
            Number of warmup rounds
        attack_fn : Callable
            Attack function to be used
        dataset_name : str
            Name of the dataset
        threshold : float, optional
            Threshold used by the attacks, by default 0.0
        to_keep : int, optional
            Number of clients to keep (i.e., to classify as "good"), by default 1
        magnitude : float, optional
            Magnitude of the Gaussian attack, by default 0.0
        sampling : int, optional
            Number of parameters to sample, by default 0
        history_dir : str, optional
            Directory where to save the parameters, by default "clients_params"
        omniscent : bool, optional
            Whether to use the omniscent attack, by default True
        """
        super().__init__(*args, **kwargs)
        self.num_malicious = num_malicious
        self.warmup_rounds = warmup_rounds
        self.attack_fn = attack_fn
        self.sampling = sampling
        self.aggregated_parameters: List = []
        self.params_indexes: List = []
        self.history_dir = history_dir
        self.dataset_name = dataset_name
        self.magnitude = magnitude
        self.threshold = threshold
        self.to_keep = to_keep
        self.omniscent = omniscent
        self.malicious_lst: List = []
        self.confusion_matrix = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        self.clients_state: Dict[str, bool] = {}
        self.good_clients_idx: List[int] = []
        self.malicious_clients_idx: List[int] = []

    # pylint: disable=too-many-locals
    def fit(self, num_rounds, timeout):
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(
            timeout=timeout, server_round=num_rounds
        )
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)

        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            res[1]["TP"] = 0
            res[1]["TN"] = 0
            res[1]["FP"] = 0
            res[1]["FN"] = 0
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
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
                # Update confusion matrix
                if current_round > self.warmup_rounds:
                    self.confusion_matrix = update_confusion_matrix(
                        self.confusion_matrix,
                        self.clients_state,
                        self.malicious_clients_idx,
                        self.good_clients_idx,
                    )

                for key, val in self.confusion_matrix.items():
                    metrics_cen[key] = val

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

    # pylint: disable-msg=R0915
    def fit_round(
        self,
        server_round,
        timeout,
    ):
        # pylint: disable-msg=R0912
        """Perform a single round of federated learning."""
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

        # Randomly decide which client is malicious
        size = self.num_malicious
        if server_round <= self.warmup_rounds:
            size = 0
        log(INFO, "Selecting %s malicious clients", size)
        self.malicious_lst = np.random.choice(
            [proxy.cid for proxy, _ in client_instructions], size=size, replace=False
        )

        # Create dict clients_state to keep track of malicious clients
        # and send the information to the clients
        clients_state = {}
        for _, (proxy, ins) in enumerate(client_instructions):
            clients_state[proxy.cid] = False
            ins.config["malicious"] = False
            if proxy.cid in self.malicious_lst:
                clients_state[proxy.cid] = True
                ins.config["malicious"] = True

        # Sort clients states
        clients_state = {k: clients_state[k] for k in sorted(clients_state)}
        log(
            DEBUG,
            "fit_round %s: malicious clients selected %s, clients_state %s",
            server_round,
            self.malicious_lst,
            clients_state,
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )
        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Save parameters of each client as time series
        ordered_results = [0 for _ in range(len(results))]
        for proxy, fitres in results:
            params = flatten_params(parameters_to_ndarrays(fitres.parameters))
            if self.sampling > 0:
                # if the sampling number is greater than the number of
                # parameters, just sample all of them
                self.sampling = min(self.sampling, len(params))
                if len(self.params_indexes) == 0:
                    # Sample a random subset of parameters
                    self.params_indexes = np.random.randint(
                        0, len(params), size=self.sampling
                    )

                params = params[self.params_indexes]

            save_params(params, fitres.metrics["cid"], params_dir=self.history_dir)

            # Re-arrange results in the same order as clients' cids impose
            ordered_results[int(fitres.metrics["cid"])] = (proxy, fitres)

        log(INFO, "Clients state: %s", clients_state)

        # Initialize aggregated_parameters if it is the first round
        if self.aggregated_parameters == []:
            for key, val in clients_state.items():
                if val is False:
                    self.aggregated_parameters = parameters_to_ndarrays(
                        ordered_results[int(key)][1].parameters
                    )
                    break

        # Apply attack function
        # the server simulates an attacker that controls a fraction of the clients
        if self.attack_fn is not None and server_round > self.warmup_rounds:
            log(INFO, "Applying attack function")
            results, _ = self.attack_fn(
                ordered_results,
                clients_state,
                omniscent=self.omniscent,
                magnitude=self.magnitude,
                w_re=self.aggregated_parameters,
                threshold=self.threshold,
                d=len(self.aggregated_parameters),
                dataset_name=self.dataset_name,
                to_keep=self.to_keep,
                malicious_num=self.num_malicious,
                num_layers=len(self.aggregated_parameters),
            )

            # Update saved parameters time series after the attack
            for _, fitres in results:
                if clients_state[fitres.metrics["cid"]]:
                    if self.sampling > 0:
                        params = flatten_params(
                            parameters_to_ndarrays(fitres.parameters)
                        )[self.params_indexes]
                    else:
                        params = flatten_params(
                            parameters_to_ndarrays(fitres.parameters)
                        )
                    log(
                        INFO,
                        "Saving parameters of client %s with shape %s after the attack",
                        fitres.metrics["cid"],
                        params.shape,
                    )
                    save_params(
                        params,
                        fitres.metrics["cid"],
                        params_dir=self.history_dir,
                        remove_last=True,
                    )
        else:
            results = ordered_results

        # Aggregate training results
        log(INFO, "fit_round - Aggregating training results")
        good_clients_idx = []
        malicious_clients_idx = []
        aggregated_result = self.strategy.aggregate_fit(server_round, results, failures)
        if isinstance(self.strategy, Flanders):
            parameters_aggregated, metrics_aggregated = aggregated_result
            malicious_clients_idx = metrics_aggregated["malicious_clients_idx"]
            good_clients_idx = metrics_aggregated["good_clients_idx"]

            log(INFO, "Malicious clients: %s", malicious_clients_idx)

            log(INFO, "clients_state: %s", clients_state)

            # For clients detected as malicious, replace the last params in
            # their history with tha current global model, otherwise the
            # forecasting in next round won't be reliable (see the paper for
            # more details)
            if server_round > self.warmup_rounds:
                log(INFO, "Saving parameters of clients")
                for idx in malicious_clients_idx:
                    if self.sampling > 0:
                        new_params = flatten_params(
                            parameters_to_ndarrays(parameters_aggregated)
                        )[self.params_indexes]
                    else:
                        new_params = flatten_params(
                            parameters_to_ndarrays(parameters_aggregated)
                        )

                    log(
                        INFO,
                        "Saving parameters of client %s with shape %s",
                        idx,
                        new_params.shape,
                    )
                    save_params(
                        new_params,
                        idx,
                        params_dir=self.history_dir,
                        remove_last=True,
                        rrl=False,
                    )
        else:
            # Aggregate training results
            log(INFO, "fit_round - Aggregating training results")
            parameters_aggregated, metrics_aggregated = aggregated_result

        self.clients_state = clients_state
        self.good_clients_idx = good_clients_idx
        self.malicious_clients_idx = malicious_clients_idx
        return parameters_aggregated, metrics_aggregated, (results, failures)
