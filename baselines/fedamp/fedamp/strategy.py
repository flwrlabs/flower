"""FedAMP: Personalized Cross-Silo Federated Learning on Non-IID Data."""

import copy
import math
from collections.abc import Callable, Iterable
from logging import INFO

import numpy as np
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
)
from flwr.common import Array, log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import sample_nodes


class FedAMP(FedAvg):
    """
    FedAMP strategy class inherits FedAvg.
    Implementation inspired by:
    https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/serveramp.py.
    """

    def __init__(
        self,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        alphaK: float = 1,
        sigma: float = 1,
        weighted_by_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
        aggregatedrecord_key: str = "mu",
        train_metrics_aggr_fn: (
            Callable[[list[RecordDict], str], MetricRecord] | None
        ) = None,
        evaluate_metrics_aggr_fn: (
            Callable[[list[RecordDict], str], MetricRecord] | None
        ) = None,
    ) -> None:

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

        self.aggregatedrecord_key = aggregatedrecord_key
        # these values are defined in pyproject.toml file, and are passes via run_config
        self.alphaK = alphaK
        self.sigma = sigma

        # store the initial arrays
        self.initial_arrays: ArrayRecord | None = None

        # set of all client arrays
        self.client_arrays: dict[int, ArrayRecord] = {}

        # set of all client ids
        self.client_ids: set = set()

        # set of prev round clients
        self.prev_round_clients: list[int] = []

    def _construct_messages(
        self, records: dict[int, RecordDict], node_ids: list[int], message_type: str
    ) -> Iterable[Message]:
        """Construct N Messages carrying the client specific RecordDict payload."""
        messages = []
        for node_id in node_ids:  # one message for each node
            message = Message(
                content=records[node_id],
                message_type=message_type,
                dst_node_id=node_id,
            )
            messages.append(message)
        return messages

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        # Do not configure federated train if fraction_train is 0.
        if self.fraction_train == 0.0:
            return []
        # Sample nodes
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
        log(
            INFO,
            "configure_train: Sampled %s nodes (out of %s)",
            len(node_ids),
            len(num_total),
        )
        # Always inject current server round
        config["server-round"] = server_round

        # for all these node_ids get their latest model,
        # if not exist then replace with the initial model
        curr_round_client_arrays: dict[int, ArrayRecord] = {}
        for node_id in node_ids:
            # For clients that do not yet exist,
            # we will set the self.client_weights[node_id] in aggregate_fn
            curr_round_client_arrays[node_id] = self.client_arrays.get(
                node_id, self.initial_arrays
            )

        # for FedAMP: other than personalized client model, we want to send the
        # partial aggregated model `mu` and the self weight (coef_self)
        # which will be used to aggregate on client side
        # this uses the curr_round_client_arrays, self.prev_round_clients
        # to calculate sim weights
        partial_aggregated_arrays = self._compute_aggregated_arrays(
            curr_round_client_arrays
        )

        # Construct messages
        records: dict[int, RecordDict] = {}
        for node_id in node_ids:
            conf = copy.deepcopy(config)
            # 1st index is the coef_self
            conf["coef_self"] = partial_aggregated_arrays[node_id][1]
            records[node_id] = RecordDict(
                {
                    self.arrayrecord_key: curr_round_client_arrays[node_id],
                    # 0th index is the array records
                    self.aggregatedrecord_key: partial_aggregated_arrays[node_id][0],
                    self.configrecord_key: conf,
                }
            )

        return self._construct_messages(records, node_ids, MessageType.TRAIN)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        # We don't aggregate models on the server for FedAMP, so arrays will be None
        arrays, metrics = None, None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]

            #  for FedAMP: no need to aggregate any client arrays,
            #  instead update the self.client_arrays and
            #  self.client_ids and self.prev_round_clients

            # clear prev round clients and store the clients for this round
            self.prev_round_clients.clear()

            for msg in valid_replies:
                self.client_arrays[msg.metadata.src_node_id] = (
                    msg.content.array_records[self.arrayrecord_key]
                )
                if msg.metadata.src_node_id not in self.client_ids:
                    self.client_ids.add(msg.metadata.src_node_id)
                self.prev_round_clients.append(msg.metadata.src_node_id)

            # Aggregate MetricRecords as usual
            metrics = self.train_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
            )

        return arrays, metrics

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated evaluation."""
        # Do not configure federated evaluation if fraction_evaluate is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Sample nodes
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_evaluate)
        sample_size = max(num_nodes, self.min_evaluate_nodes)
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
        log(
            INFO,
            "configure_evaluate: Sampled %s nodes (out of %s)",
            len(node_ids),
            len(num_total),
        )

        # Always inject current server round
        config["server-round"] = server_round

        # Construct messages
        records: dict[int, RecordDict] = {
            node_id: RecordDict(
                {
                    self.arrayrecord_key: self.client_arrays.get(
                        node_id, self.initial_arrays
                    ),
                    self.configrecord_key: config,
                }
            )
            for node_id in node_ids
        }

        return self._construct_messages(records, node_ids, MessageType.EVALUATE)

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> MetricRecord | None:
        """Aggregate MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=False)

        metrics = None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]

            # Aggregate MetricRecords
            metrics = self.evaluate_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
            )
        return metrics

    def _compute_aggregated_arrays(
        self, curr_round_client_arrays: dict[int, ArrayRecord]
    ) -> dict[int, tuple[ArrayRecord, float]]:
        """Compute personalized attentive message (mu) for each client."""
        # use the curr_round_model_arrays, prev_round_clients to calculate the sims
        # return the partial_aggregated_arrays, and coef_sim

        partial_aggregated_arrays: dict[int, tuple[ArrayRecord, float]] = {}

        if len(self.prev_round_clients) > 0:
            for client_id, client_arrays in curr_round_client_arrays.items():

                mu_np = {
                    key: np.zeros_like(value.numpy())  # clear the aggregate
                    for key, value in (
                        self.initial_arrays.items()
                        if self.initial_arrays is not None
                        else {}
                    )
                }

                coef = np.zeros(
                    len(self.prev_round_clients)
                )  # this should be number of selected clients
                prev_arrays_list = []
                for j, prev_client_id in enumerate(self.prev_round_clients):
                    prev_client_arrays = self.client_arrays[
                        prev_client_id
                    ]  # this will always exist as, if the client was selected in
                    # prev round then it would have been added to the clients set
                    prev_arrays_list.append(prev_client_arrays)

                    if client_id != prev_client_id:
                        # Flatten once
                        weights_i = np.concatenate(
                            [p.numpy().flatten() for p in client_arrays.values()],
                            axis=0,
                        )
                        weights_j = np.concatenate(
                            [p.numpy().flatten() for p in prev_client_arrays.values()],
                            axis=0,
                        )
                        dist_sq = np.dot(weights_i - weights_j, weights_i - weights_j)
                        coef[j] = self.alphaK * self.exp_sim(dist_sq, self.sigma)
                    else:
                        coef[j] = 0

                coef_self = 1 - np.sum(coef)

                for j, prev_client_arrays in enumerate(prev_arrays_list):
                    if coef[j] > 0:
                        for key in mu_np.keys():
                            if prev_client_arrays is not None:
                                mu_np[key] += coef[j] * prev_client_arrays[key].numpy()

                mu_record = ArrayRecord({k: Array(v) for k, v in mu_np.items()})
                partial_aggregated_arrays[client_id] = (mu_record, coef_self)
        else:
            # For first round no personalization, complete local training...
            for client_id in curr_round_client_arrays.keys():
                mu_np = {}
                for key, value in (
                    self.initial_arrays.items()
                    if self.initial_arrays is not None
                    else {}.items()
                ):
                    mu_np[key] = np.zeros_like(value.numpy())

                mu_record = ArrayRecord({k: Array(v) for k, v in mu_np.items()})
                partial_aggregated_arrays[client_id] = (mu_record, 1.0)

        return partial_aggregated_arrays

    @staticmethod
    def exp_sim(x, sigma):
        """Exponential similarity kernel."""
        return math.exp(-x / sigma) / sigma

    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: ConfigRecord | None = None,
        evaluate_config: ConfigRecord | None = None,
        evaluate_fn: Callable[[int, ArrayRecord], MetricRecord | None] | None = None,
    ):
        """Override the base strategy start method to store initial arrays."""
        self.initial_arrays = copy.deepcopy(
            initial_arrays
        )  # create a copy incase later initial_arrays gets changed inplace.
        return super().start(
            grid,
            initial_arrays,
            num_rounds,
            timeout,
            train_config,
            evaluate_config,
            evaluate_fn,
        )
