"""TraceFL Strategy extending FedAvg for provenance analysis."""

import logging
import traceback
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from flwr.common import MetricRecord
from flwr.common.logger import log
from flwr.serverapp.strategy import FedAvg

from .fl_provenance import FlowerProvenance
from .model import initialize_model
from .results_logging import ExperimentResultLogger


class TraceFLStrategy(FedAvg):
    """Custom Flower strategy that extends FedAvg with TraceFL provenance analysis."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        weighted_by_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
        train_metrics_aggr_fn: Callable[[list, str], MetricRecord] | None = None,
        evaluate_metrics_aggr_fn: Callable[[list, str], MetricRecord] | None = None,
        # TraceFL-specific parameters
        provenance_rounds: list[int] | None = None,
        enable_beta: bool = True,
        client_weights_normalization: bool = True,
        cfg: Any | None = None,
        output_dir: Any | None = None,
    ) -> None:
        """Initialize TraceFL strategy.

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. Defaults to 1.0.
        fraction_evaluate : float, optional
            Fraction of clients used during evaluation. Defaults to 1.0.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during evaluation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[callable], optional
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Optional[callable], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Optional[callable], optional
            Function used to configure evaluation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Optional[bytes], optional
            Initial global model parameters. Defaults to None.
        fit_metrics_aggregation_fn : Optional[callable], optional
            Metrics aggregation function for fit. Defaults to None.
        evaluate_metrics_aggregation_fn : Optional[callable], optional
            Metrics aggregation function for evaluate. Defaults to None.
        provenance_rounds : Optional[List[int]], optional
            List of rounds to run provenance analysis. Defaults to None.
        enable_beta : bool, optional
            Enable layer importance weighting. Defaults to True.
        client_weights_normalization : bool, optional
            Normalize client contributions. Defaults to True.
        cfg : Optional[object], optional
            Configuration object. Defaults to None.
        """
        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=min_train_nodes,
            min_evaluate_nodes=min_evaluate_nodes,
            min_available_nodes=min_available_nodes,
            weighted_by_key=weighted_by_key,
            arrayrecord_key=arrayrecord_key,
            configrecord_key=configrecord_key,
            train_metrics_aggr_fn=train_metrics_aggr_fn,
            evaluate_metrics_aggr_fn=evaluate_metrics_aggr_fn,
        )

        # TraceFL-specific attributes
        self.provenance_rounds = provenance_rounds or []
        self.enable_beta = enable_beta
        self.client_weights_normalization = client_weights_normalization
        self.cfg = cfg
        self.output_dir = output_dir

        # Storage for client models and metadata (persists across rounds)
        #
        # self.client_models: Stores client state dicts for provenance analysis
        #   Structure: {round_id: {client_id: model_state_dict}}
        #   Example: {1: {0: {...}, 1: {...}}, 2: {0: {...}, 2: {...}}}
        #   Used by: NeuronProvenance to trace neuron-level contributions
        #
        # self.client_num_examples: Number of training examples per client per round
        #   Structure: {round_id: {client_id: num_examples}}
        #   Used by: Weight normalization in provenance analysis (p_k = n_k / sum(n_k))
        #
        # self.client2class: Label distribution for each client (all rounds combined)
        #   Structure: {client_id: {label: count}}
        #   Example: {0: {0: 100, 1: 50}, 1: {2: 80, 3: 120}}
        #   Used by: Computing "responsible clients" for localization accuracy
        #
        self.client_models: dict[int, dict[int, dict[str, Any]]] = {}
        self.client_num_examples: dict[int, dict[int, int]] = {}
        self.server_test_data: Any = None
        self.client2class: dict[int, dict[str, int]] = {}
        self._result_logger: Any = None

        log(
            logging.INFO,
            "TraceFL Strategy initialized with provenance rounds: %s",
            self.provenance_rounds,
        )

    def aggregate_train(
        self,
        server_round: int,
        replies,
    ):
        """Aggregate train results and store client models for provenance analysis.

        IMPORTANT: This method is called by FedAvg.start() and may be wrapped
        by TraceFLWithDP. The wrapping order is:
        1. Client replies arrive
        2. This method is called (by DP wrapper if enabled)
        3. _store_client_models extracts ORIGINAL client models
        4. super().aggregate_train() does weighted averaging
        5. DP wrapper (if enabled) clips and adds noise to the result
        6. Provenance analysis runs with ORIGINAL client models (not noisy)

        This design ensures provenance analysis uses unmodified client models
        while the global model still receives DP protection.
        """
        # Store client data BEFORE calling parent (which may be wrapped by DP)
        # Must happen first to capture original models before DP noise is added
        replies_list = list(replies)

        # Extract and store client models from original replies (no DP applied yet)
        self._store_client_models(server_round, replies_list)

        # Call parent aggregate_train with the list (FedAvg weighted averaging)
        arrays, metrics = super().aggregate_train(server_round, replies_list)

        # Trigger provenance analysis if this round is in provenance_rounds
        # Uses stored ORIGINAL client models (not the DP-protected global model)
        if server_round in self.provenance_rounds:
            self._run_provenance_analysis(server_round, arrays)

        return arrays, metrics

    def _store_client_models(self, server_round: int, replies_list: list) -> None:
        """Extract and store client models from replies for provenance analysis.

        This must be called BEFORE the parent's aggregate_train processes replies,
        especially when using DP wrapper which modifies reply contents.
        """
        if not replies_list:
            return

        log(
            logging.INFO,
            "Processing %s replies for round %s",
            len(replies_list),
            server_round,
        )

        self.client_models[server_round] = {}
        self.client_num_examples[server_round] = {}

        # Extract client models from Message objects
        for msg in replies_list:
            if not msg.has_error():
                # Extract actual client ID from metrics (not sequential mapping)
                metric_content = msg.content.get("metrics")
                if metric_content is not None:
                    # Get actual client ID from metrics
                    client_id = metric_content.get("client_id")
                    if client_id is None:
                        logging.warning(
                            "No client_id found in metrics, skipping message"
                        )
                        continue

                    num_examples = metric_content.get(self.weighted_by_key, 0)

                    # Extract ArrayRecord (model weights)
                    arrayrecord = msg.content.get(self.arrayrecord_key)
                    if arrayrecord:
                        # Convert ArrayRecord to PyTorch state_dict
                        state_dict = arrayrecord.to_torch_state_dict()
                        self.client_models[server_round][client_id] = state_dict

                    self.client_num_examples[server_round][client_id] = num_examples

                    logging.debug(
                        "Stored model for client %s with %s examples",
                        client_id,
                        num_examples,
                    )

        log(
            logging.INFO,
            "Stored models for %s clients in round %s",
            len(self.client_models[server_round]),
            server_round,
        )

    def _run_provenance_analysis(self, server_round: int, global_arrays):
        """Run provenance analysis for the current round."""
        if not self.cfg or not self.server_test_data:
            log(
                logging.WARNING,
                "Cannot run provenance analysis: missing config or test data",
            )
            return

        try:
            # Create global model from aggregated ArrayRecord
            model_dict = initialize_model(
                self.cfg.data_dist.model_name, self.cfg.data_dist
            )
            global_model = model_dict["model"]

            # Load global parameters from ArrayRecord
            if global_arrays:
                state_dict = global_arrays.to_torch_state_dict()
                # Convert numpy arrays to tensors
                state_dict = {
                    k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                    for k, v in state_dict.items()
                }
                global_model.load_state_dict(state_dict)

            # Create client models
            client2model = {}
            for client_id, state_dict in self.client_models[server_round].items():
                client_model_dict = initialize_model(
                    self.cfg.data_dist.model_name, self.cfg.data_dist
                )
                client_model = client_model_dict["model"]
                # Convert numpy arrays to tensors
                state_dict_tensors = {
                    k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                    for k, v in state_dict.items()
                }
                client_model.load_state_dict(state_dict_tensors)
                client2model[client_id] = client_model

            # Initialize provenance analysis
            provenance = FlowerProvenance(
                cfg=self.cfg,
                round_id=server_round,
                server_test_data=self.server_test_data,
                global_model=global_model,
                client2model=client2model,
                client2num_examples=self.client_num_examples[server_round],
                client2class=self.client2class,
            )

            # Calculate client contributions
            results = provenance.calculate_client_contributions()

            if "error" not in results:
                log(
                    logging.INFO,
                    "Provenance analysis completed for round %s",
                    server_round,
                )
                log(
                    logging.INFO,
                    "Samples analyzed: %s",
                    results["samples_analyzed"],
                )
                log(
                    logging.INFO,
                    "Top contributor: Client %s",
                    results["top_contributor"],
                )

                # Persist results for downstream plotting utilities
                try:
                    result_logger = self._get_result_logger()
                    result_logger.record_round(server_round, results)
                    log(
                        logging.INFO,
                        "Stored provenance metrics in %s",
                        result_logger.file_path,
                    )
                except (
                    OSError,
                    ValueError,
                    AttributeError,
                ) as exc:  # pragma: no cover - defensive logging
                    log(
                        logging.WARNING,
                        "Failed to persist provenance metrics: %s",
                        exc,
                    )
            else:
                log(
                    logging.ERROR,
                    "Provenance analysis failed: %s",
                    results["error"],
                )

        except (ValueError, RuntimeError, AttributeError, KeyError) as e:
            log(
                logging.ERROR,
                "Error in provenance analysis: %s",
                e,
            )
            log(
                logging.ERROR,
                "Traceback:\n%s",
                traceback.format_exc(),
            )
            log(
                logging.ERROR,
                "Error type: %s",
                type(e),
            )
            log(
                logging.ERROR,
                "Error args: %s",
                e.args,
            )

    def set_server_test_data(self, test_data):
        """Set server test data for provenance analysis."""
        self.server_test_data = test_data
        log(
            logging.INFO,
            "Server test data set: %s samples",
            len(test_data),
        )

    def set_client2class(self, client2class):
        """Set client-to-class mappings for provenance analysis."""
        normalized = {}
        for cid, label_counts in client2class.items():
            try:
                client_id = int(cid)
            except (TypeError, ValueError):
                client_id = cid

            normalized_counts = {}
            for label, count in (label_counts or {}).items():
                if isinstance(count, torch.Tensor):
                    count = int(count.item())
                try:
                    count_int = int(count)
                except (TypeError, ValueError):
                    continue
                label_str = str(label.item() if hasattr(label, "item") else label)
                normalized_counts[label_str] = count_int

            normalized[client_id] = normalized_counts

        self.client2class = normalized
        log(
            logging.INFO,
            "Client2class mapping set for %s clients",
            len(self.client2class),
        )

    def configure_fit(
        self, _server_round: int, _parameters: bytes, client_manager
    ) -> list[tuple[int, dict[str, Any]]]:
        """Configure the next round of training."""
        # Get base configuration from parent class
        config: list[tuple[int, dict[str, Any]]] = []
        for client_id in client_manager.sample(
            num_clients=client_manager.num_available(),
            min_num_clients=self.min_train_nodes,
        ):
            config.append((client_id, {}))

        # Add TraceFL-specific configuration
        if self.cfg:
            for i, (client_id, client_config) in enumerate(config):
                config[i] = (
                    client_id,
                    {
                        **client_config,
                        "tracefl": {
                            "model_name": self.cfg.data_dist.model_name,
                            "model_architecture": self.cfg.data_dist.model_architecture,
                            "num_classes": self.cfg.data_dist.num_classes,
                            "channels": self.cfg.data_dist.channels,
                        },
                    },
                )

        return config

    def _get_result_logger(self):
        if self._result_logger is None and self.cfg is not None:
            self._result_logger = ExperimentResultLogger(
                cfg=self.cfg,
                output_dir=self.output_dir,
            )
        return self._result_logger
