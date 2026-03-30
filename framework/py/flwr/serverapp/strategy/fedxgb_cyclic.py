# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Flower message-based FedXgbCyclic strategy."""


from collections.abc import Callable, Iterable
from logging import INFO
from typing import cast

from flwr.app import MessageType
from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MetricRecord,
    RecordDict,
    log,
)
from flwr.server import Grid

from .fedavg import FedAvg
from .strategy_utils import sample_nodes


# pylint: disable=line-too-long
class FedXgbCyclic(FedAvg):
    """Configurable FedXgbCyclic strategy implementation.

    Parameters
    ----------
    fraction_train : float (default: 1.0)
        Fraction of nodes used during training. In case `min_train_nodes`
        is larger than `fraction_train * total_connected_nodes`, `min_train_nodes`
        will still be sampled.
    fraction_evaluate : float (default: 1.0)
        Fraction of nodes used during validation. In case `min_evaluate_nodes`
        is larger than `fraction_evaluate * total_connected_nodes`,
        `min_evaluate_nodes` will still be sampled.
    min_available_nodes : int (default: 2)
        Minimum number of total nodes in the system.
    weighted_by_key : str (default: "num-examples")
        The key within each MetricRecord whose value is used as the weight when
        computing weighted averages for MetricRecords.
    arrayrecord_key : str (default: "arrays")
        Key used to store the ArrayRecord when constructing Messages.
    configrecord_key : str (default: "config")
        Key used to store the ConfigRecord when constructing Messages.
    train_metrics_aggr_fn : Optional[callable] (default: None)
        Function with signature (list[RecordDict], str) -> MetricRecord,
        used to aggregate MetricRecords from training round replies.
        If `None`, defaults to `aggregate_metricrecords`, which performs a weighted
        average using the provided weight factor key.
    evaluate_metrics_aggr_fn : Optional[callable] (default: None)
        Function with signature (list[RecordDict], str) -> MetricRecord,
        used to aggregate MetricRecords from training round replies.
        If `None`, defaults to `aggregate_metricrecords`, which performs a weighted
        average using the provided weight factor key.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_available_nodes: int = 2,
        weighted_by_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
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
            min_train_nodes=2,
            min_evaluate_nodes=2,
            min_available_nodes=min_available_nodes,
            weighted_by_key=weighted_by_key,
            arrayrecord_key=arrayrecord_key,
            configrecord_key=configrecord_key,
            train_metrics_aggr_fn=train_metrics_aggr_fn,
            evaluate_metrics_aggr_fn=evaluate_metrics_aggr_fn,
        )

        self.registered_nodes: dict[int, int] = {}

        if fraction_train not in (0.0, 1.0):
            raise ValueError(
                "fraction_train can only be set to 1.0 or 0.0 for FedXgbCyclic."
            )
        if fraction_evaluate not in (0.0, 1.0):
            raise ValueError(
                "fraction_evaluate can only be set to 1.0 or 0.0 for FedXgbCyclic."
            )

    def _reorder_nodes(self, node_ids: list[int]) -> list[int]:
        """Re-order node ids based on registered nodes.

        Each node ID is assigned a persistent index in `self.registered_nodes`
        the first time it appears. The input list is then reordered according
        to these stored indices, and the result is compacted into ascending
        order (1..N) for the current call.
        """
        # Assign new indices to unknown nodes
        next_index = max(self.registered_nodes.values(), default=0) + 1
        for nid in node_ids:
            if nid not in self.registered_nodes:
                self.registered_nodes[nid] = next_index
                next_index += 1

        # Sort node_ids by their stored indices
        sorted_by_index = sorted(node_ids, key=lambda x: self.registered_nodes[x])

        # Compact re-map of indices just for this output list
        unique_indices = sorted(self.registered_nodes[nid] for nid in sorted_by_index)
        remap = {old: new for new, old in enumerate(unique_indices, start=1)}

        # Build the result list ordered by compact indices
        result_list = [
            nid
            for _, nid in sorted(
                (remap[self.registered_nodes[nid]], nid) for nid in sorted_by_index
            )
        ]
        return result_list

    def _make_sampling(
        self, grid: Grid, server_round: int, configure_type: str
    ) -> list[int]:
        """Sample nodes using the Grid."""
        # Sample nodes
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)
        node_ids, _ = sample_nodes(grid, self.min_available_nodes, sample_size)

        # Re-order node_ids
        node_ids = self._reorder_nodes(node_ids)

        # Sample the clients sequentially given server_round
        sampled_idx = (server_round - 1) % len(node_ids)
        sampled_node_id = [node_ids[sampled_idx]]

        log(
            INFO,
            f"{configure_type}: Sampled %s nodes (out of %s)",
            len(sampled_node_id),
            len(node_ids),
        )
        return sampled_node_id

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        # Sample one node
        sampled_node_id = self._make_sampling(grid, server_round, "configure_train")

        # Always inject current server round
        config["server-round"] = server_round

        # Construct messages
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, sampled_node_id, MessageType.TRAIN)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        arrays, metrics = None, None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]
            array_record_key = next(iter(reply_contents[0].array_records.keys()))

            # Fetch the client model from current round as global model
            arrays = cast(ArrayRecord, reply_contents[0][array_record_key])

            # Aggregate MetricRecords
            metrics = self.train_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
            )
        return arrays, metrics

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated evaluation."""
        # Sample one node
        sampled_node_id = self._make_sampling(grid, server_round, "configure_evaluate")

        # Always inject current server round
        config["server-round"] = server_round

        # Construct messages
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, sampled_node_id, MessageType.EVALUATE)
