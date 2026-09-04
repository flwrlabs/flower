#
# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower message-based Flad strategy."""

import io
import math
import time
from collections.abc import Callable, Iterable
from logging import INFO, WARNING
from typing import Any, cast

import numpy as np
from flwr.app import (
    Array,
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
)
from flwr.common import NDArray, log
from flwr.serverapp.exception import InconsistentMessageReplies
from flwr.serverapp.grid import Grid
from flwr.serverapp.strategy.result import Result
from flwr.serverapp.strategy.strategy import Strategy
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info


# pylint: disable=too-many-instance-attributes
class Flad(Strategy):
    """FLAD: Adaptive Federated Learning strategy for DDoS attack detection.

    Implementation based on https://arxiv.org/abs/2205.06661

    Parameters
    ----------
    client_names : str
        Comma-separated list of client names. Each client will be created
        with the specified name.
    mapping_clients_timeout : float (default: 3600)
        Timeout in seconds for mapping client names to node IDs in the grid.
    arrayrecord_key : str (default: "arrays")
        Key used to store the ArrayRecord when constructing Messages.
    configrecord_key : str (default: "config")
        Key used to store the ConfigRecord when constructing Messages.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        client_names: str,
        mapping_clients_timeout: float = 3600.0,
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
    ) -> None:
        self.arrayrecord_key = arrayrecord_key
        self.configrecord_key = configrecord_key
        clients: list[dict[str, Any]] = []
        for name in client_names.split(","):
            client: dict[str, Any] = {}
            client["name"] = name
            client["epochs"] = 0
            client["steps_per_epoch"] = 0
            client["f1_score"] = 0
            client[self.arrayrecord_key] = None
            clients.append(client)

        self.clients = clients
        self.last_round_average_f1 = 1.0
        self.best_average_f1 = 0.0
        self.round_participants: list[dict[str, Any]] = []
        self.is_mapped = False
        self.mapping_clients_timeout = mapping_clients_timeout

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> Clients participating to the federation:")
        for client in self.clients:
            log(INFO, "\t│\t└── %s", client["name"])
        log(INFO, "\t└──> Keys in records:")
        log(INFO, "\t\t├── ArrayRecord key: '%s'", self.arrayrecord_key)
        log(INFO, "\t\t└── ConfigRecord key: '%s'", self.configrecord_key)

    def _map_clients(
        self,
        grid: Grid,
    ) -> None:
        """Query clients to map client names to node IDs in the grid."""
        # Wait until all clients are online
        while len(node_ids := list(grid.get_node_ids())) < len(self.clients):
            log(
                INFO,
                "Waiting for nodes to connect: %d connected (target: %d).",
                len(node_ids),
                len(self.clients),
            )
            time.sleep(1)

        msgs = []
        log(INFO, "Mapping clients to node IDs in the grid.")
        for nid in node_ids:
            request = ConfigRecord({"request": "client_info"})
            msgs.append(
                Message(
                    content=RecordDict({self.configrecord_key: request}),
                    message_type="query.info",
                    dst_node_id=nid,
                )
            )

        replies = grid.send_and_receive(msgs, timeout=self.mapping_clients_timeout)
        replies = list(replies)
        if len(replies) < len(node_ids):
            raise InconsistentMessageReplies(
                reason=(
                    f"Expected replies from {len(node_ids)} nodes, "
                    f"but received {len(replies)}."
                )
            )

        for reply in replies:
            if reply.has_error():
                raise InconsistentMessageReplies(
                    reason=(
                        "Reply from client with node ID "
                        f"{reply.metadata.src_node_id} has error: {reply.error}."
                    )
                )
            client = next(
                (
                    c
                    for c in self.clients
                    if c["name"] == reply.content[self.configrecord_key]["name"]
                ),
                None,
            )
            if client is None:
                raise InconsistentMessageReplies(
                    reason=(
                        "Received evaluation message from unknown client with node ID "
                        f"{reply.metadata.src_node_id}"
                    )
                )
            client["id"] = reply.metadata.src_node_id

    def _is_mapping_valid(self, grid: Grid) -> bool:
        """Check whether every previously-mapped client is still connected.

        Returns False if any client has disconnected and reconnected with a
        new node ID since the last mapping.
        """
        current_node_ids = set(grid.get_node_ids())
        return all(client["id"] in current_node_ids for client in self.clients)

    def _validate_train_config(self, train_config: ConfigRecord) -> None:
        """Validate that train_config has all required keys  with the
        correct (integer) types."""
        required_int_keys = (
            "min_epochs",
            "max_epochs",
            "min_steps",
            "max_steps",
            "patience",
        )
        for key in required_int_keys:
            if key not in train_config:
                raise ValueError(f"Missing required key '{key}' in train_config.")
            if not isinstance(train_config[key], int):
                raise TypeError(f"'{key}' in train_config must be an integer.")

    def _select_clients(
        self,
        all_clients: bool = False,
    ) -> list[dict[str, Any]]:
        """Select the clients participating in this round based on F1 Score."""
        selected_clients: list[dict[str, Any]] = []
        for client in self.clients:
            if all_clients or client["f1_score"] <= self.last_round_average_f1:
                selected_clients.append(client)
        return selected_clients

    def _scale_linear_bycolumn(
        self,
        rawpoints: NDArray,
        mins: np.floating[Any],
        maxs: np.floating[Any],
        high: float = 1.0,
        low: float = 0.0,
    ) -> NDArray:
        rng = maxs - mins
        result: NDArray = high - (((high - low) * (maxs - rawpoints)) / rng)
        return result

    def _update_client_training_parameters(
        self,
        parameter: str,
        min_value: int,
        max_value: int,
    ) -> None:
        """Compute per-client training parameters from F1 Score."""
        f1_list = []

        for client in self.round_participants:
            f1_list.append(client["f1_score"])

        if len(set(f1_list)) > 1:
            min_f1_value = min(f1_list)
            max_value = max(min_value + 1, math.ceil(max_value * (1 - min_f1_value)))
            value_array = (
                max_value
                + min_value
                - self._scale_linear_bycolumn(
                    np.asarray(f1_list),
                    np.min(f1_list),
                    np.max(f1_list),
                    high=float(max_value),
                    low=min_value,
                )
            )
        else:
            value_array = np.asarray([max_value] * len(self.round_participants))

        for client in self.round_participants:
            client[parameter] = int(value_array[self.round_participants.index(client)])

    def _construct_messages(
        self,
        record: RecordDict,
        message_type: str,
    ) -> Iterable[Message]:
        """Construct a different message for each participant client."""
        messages = []
        for client in self.round_participants:
            if message_type == MessageType.TRAIN:
                this_client_config = record[self.configrecord_key].copy()
                this_client_config["epochs"] = client["epochs"]
                this_client_config["steps_per_epoch"] = client["steps_per_epoch"]
                record[self.configrecord_key] = this_client_config
                client_record = RecordDict(
                    {
                        self.arrayrecord_key: record[self.arrayrecord_key],
                        self.configrecord_key: this_client_config,
                    }
                )

            else:
                client_record = record
            message = Message(
                content=client_record,
                message_type=message_type,
                dst_node_id=client["id"],
            )
            messages.append(message)

        return messages

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        # Map clients to node IDs in the grid. Clients are pets here, not cattle.
        # If a previously-mapped client has disconnected and reconnected (getting
        # a new node ID from the grid in the process), redo the mapping.
        if not self.is_mapped or not self._is_mapping_valid(grid):
            self._map_clients(grid)
            for client in self.clients:
                client[self.arrayrecord_key] = arrays
            self.is_mapped = True

        # Select participants to this round based on F1 Score
        self.round_participants = self._select_clients(all_clients=False)
        log(
            INFO,
            "configure_train: selected clients: %s",
            [client["name"] for client in self.round_participants],
        )

        # Update training parameters for the selected clients.
        # Validity of these keys is checked once, upfront, in _validate_train_config.
        minepochs = cast(int, config["min_epochs"])
        maxepochs = cast(int, config["max_epochs"])
        minsteps = cast(int, config["min_steps"])
        maxsteps = cast(int, config["max_steps"])

        self._update_client_training_parameters(
            "epochs",
            minepochs,
            maxepochs,
        )
        self._update_client_training_parameters(
            "steps_per_epoch",
            minsteps,
            maxsteps,
        )

        config_record = ConfigRecord(
            {
                "server_round": server_round,
            }
        )

        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config_record}
        )

        return self._construct_messages(record, MessageType.TRAIN)

    def _check_reply_from_client(
        self,
        reply: Message,
    ) -> dict[str, Any]:
        """Check that a reply is valid.

        Raises an exception if the reply carries an error, or if it
        comes from an unrecognized client.
        """
        # In case of error, we raise an exception. An alternative could be to
        # ignore the client and continue with the other clients.
        if reply.has_error():
            raise InconsistentMessageReplies(
                f"Error in training message from client with node ID "
                f"{reply.metadata.src_node_id}: {reply.error}"
            )
        client = next(
            (c for c in self.clients if c["id"] == reply.metadata.src_node_id), None
        )
        if client is None:
            raise InconsistentMessageReplies(
                f"Received training message from unknown client with node ID "
                f"{reply.metadata.src_node_id}"
            )

        return client

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord, MetricRecord]:
        """Aggregate model weights from all clients.

        Aggregation is unweighted — a simple average of the model weights from
        every client, not only those that participated in this round. Also
        collects (without aggregating) training metrics from all replying clients.
        """
        metric_record = MetricRecord({})
        replies = list(replies)
        log(INFO, "aggregate_train: received %d replies from clients", len(replies))
        if len(replies) < len(self.round_participants):
            log(
                WARNING,
                "aggregate_train: expected %d replies but received only %d; "
                "%d client(s) did not respond this round.",
                len(self.round_participants),
                len(replies),
                len(self.round_participants) - len(replies),
            )
        for reply in replies:
            client = self._check_reply_from_client(reply)
            client[self.arrayrecord_key] = reply.content[self.arrayrecord_key]

            # We just collect the metrics without aggregating them. A client may
            # report metrics under any number of differently-named MetricRecords.
            for metric_record_item in reply.content.metric_records.values():
                for key, value in metric_record_item.items():
                    metric_record[f"{client['name']}_{key}"] = value

        aggregated_np_arrays: dict[str, NDArray] = {}
        for client in self.clients:
            for array_key, array_value in client[self.arrayrecord_key].items():
                if array_key not in aggregated_np_arrays:
                    aggregated_np_arrays[array_key] = array_value.numpy()
                else:
                    aggregated_np_arrays[array_key] += array_value.numpy()

        for key in aggregated_np_arrays:
            aggregated_np_arrays[key] = aggregated_np_arrays[key] / len(self.clients)

        return (
            ArrayRecord(
                {k: Array(np.asarray(v)) for k, v in aggregated_np_arrays.items()}
            ),
            metric_record,
        )

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated evaluation."""
        # Select all clients for evaluation phase.
        self.round_participants = self._select_clients(all_clients=True)
        log(INFO, "configure_evaluate: on %d clients", len(self.round_participants))

        # Always inject current server round
        config["server_round"] = server_round

        # Construct messages
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, MessageType.EVALUATE)

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> MetricRecord:
        """Aggregate evaluation metrics from all clients.

        Aggregation is unweighted — a simple average of the per-client metrics.
        """
        # Collect evaluation metrics from all clients
        metric_record = MetricRecord({})

        all_clients_f1_scores = []

        replies = list(replies)

        if not replies:
            raise InconsistentMessageReplies(
                f"No evaluation replies received in round {server_round}."
            )

        if len(replies) < len(self.round_participants):
            raise InconsistentMessageReplies(
                f"aggregate_evaluate: expected {len(self.round_participants)} replies "
                f"but received only {len(replies)}; "
            )
        for reply in replies:
            client = self._check_reply_from_client(reply)
            if "f1_score" not in reply.content.metric_records["metrics"]:
                raise InconsistentMessageReplies(
                    f"f1_score not found in evaluation metrics from client with "
                    f"node ID {reply.metadata.src_node_id}"
                )
            f1_score = reply.content.metric_records["metrics"]["f1_score"]
            if not isinstance(f1_score, float):
                raise InconsistentMessageReplies(
                    f"f1_score must be a float, got {type(f1_score).__name__}, "
                    f"from client with node ID {reply.metadata.src_node_id}"
                )
            client["f1_score"] = f1_score
            all_clients_f1_scores.append(client["f1_score"])

            # Record all metrics reported by the client (including f1_score,
            # already validated above), under any number of differently-named
            # MetricRecords.
            for metric_record_item in reply.content.metric_records.values():
                for key, value in metric_record_item.items():
                    metric_record[f"{client['name']}_{key}"] = value

        # Average f1_score on all clients
        avg = np.average(all_clients_f1_scores)
        metric_record["avg_f1_score"] = float(avg)

        self.best_average_f1 = max(self.best_average_f1, avg)

        metric_record["avg_f1_score_best"] = float(self.best_average_f1)

        return metric_record

    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 0,
        timeout: float = 3600,
        train_config: ConfigRecord | None = None,
        evaluate_config: ConfigRecord | None = None,
        evaluate_fn: Callable[[int, ArrayRecord], MetricRecord | None] | None = None,
    ) -> Result:
        """Execute the Flad federated learning strategy.

        Runs the complete federated learning workflow, including training and
        evaluation. This method is customized for Flad strategy. When num_rounds
        is set to 0, the strategy will continue until a stopping condition is met,
        while if num_rounds is set to a positive integer, the strategy will continue
        for that number of rounds.

        Parameters
        ----------
        grid : Grid
            The Grid instance used to send/receive Messages from nodes executing a
            ClientApp.
        initial_arrays : ArrayRecord
            Initial model parameters (arrays) to be used for federated learning.
        num_rounds : int (default: 0)
            Number of federated learning rounds to execute if > 0, otherwise the
            strategy will continue until a stopping condition is met.
        timeout : float (default: 3600)
            Timeout in seconds for waiting for node responses.
        train_config : ConfigRecord, optional
            Configuration to be sent to nodes during training rounds.
            If unset, an empty ConfigRecord will be used.
        evaluate_config : ConfigRecord, optional
            Configuration to be sent to nodes during evaluation rounds.
            If unset, an empty ConfigRecord will be used.
        evaluate_fn : Callable[[int, ArrayRecord], Optional[MetricRecord]], optional
            Not used in this strategy. This parameter is included only for
            compatibility with the base Strategy class.

        Returns
        -------
        Results
            Results containing best model arrays, training metrics and
            evaluation metrics from all rounds.

        """
        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
        self.summary()
        log(INFO, "")

        # Initialize if None
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        # Validate train_config once, upfront, so misconfiguration fails fast
        # instead of raising deep into the round loop.
        self._validate_train_config(train_config)
        patience = cast(int, train_config["patience"])

        t_start = time.time()

        arrays = initial_arrays
        current_round = 0
        stop_counter = 0
        stop_condition = False
        # Loop until the stopping condition is met or the maximum number of rounds is
        # reached.
        while not stop_condition and (num_rounds <= 0 or current_round < num_rounds):
            current_round += 1
            log(INFO, "")
            log(INFO, "[ROUND %s]", current_round)

            # -----------------------------------------------------------------
            # --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure training round. Send messages and wait for
            # replies
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    arrays,
                    train_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate train.
            agg_arrays, train_metrics = self.aggregate_train(
                current_round, train_replies
            )

            # Update server and client models with the aggregated model weights.
            if agg_arrays is not None:
                arrays = agg_arrays

            for client in self.clients:
                client[self.arrayrecord_key] = arrays

            # Log training metrics and append to history.
            if train_metrics is not None:
                log(INFO, "\t└──> Training MetricRecord: %s", train_metrics)
                result.train_metrics_clientapp[current_round] = train_metrics

            # -----------------------------------------------------------------
            # --- EVALUATION (CLIENTAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Configure evaluation round. Send messages and wait for replies
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    arrays,
                    evaluate_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate evaluate.
            # Note: this function updates self.best_average_f1 to the max
            # of its previous value and this round's average, so we capture
            # the previous value first to detect strict improvement.
            previous_best_average_f1 = self.best_average_f1
            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                evaluate_replies,
            )

            # Log evaluation metrics and append to history.
            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Evaluation MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics

            # Calculate stopping condition and save the best model.
            self.last_round_average_f1 = cast(
                float, agg_evaluate_metrics["avg_f1_score"]
            )
            if self.last_round_average_f1 <= previous_best_average_f1:
                stop_counter += 1
            else:
                # Contrary to Flower standard strategy, we keep the best model not the
                # last one.
                result.arrays = arrays
                stop_counter = 0

            log(INFO, "Current patience counter: %d", stop_counter)

            # Check stopping condition
            stop_condition = stop_counter > patience

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        return result
