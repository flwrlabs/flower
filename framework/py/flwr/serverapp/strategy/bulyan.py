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
"""Bulyan [El Mhamdi et al., 2018] strategy.

Paper: arxiv.org/abs/1802.07927
"""


from collections.abc import Callable, Iterable
from logging import INFO, WARN
from typing import cast

import numpy as np

from flwr.common import (
    Array,
    ArrayRecord,
    Message,
    MetricRecord,
    NDArrays,
    RecordDict,
    log,
)

from .fedavg import FedAvg
from .multikrum import select_multikrum


# pylint: disable=too-many-instance-attributes
class Bulyan(FedAvg):
    """Bulyan strategy.

    Implementation based on https://arxiv.org/abs/1802.07927.

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
    min_train_nodes : int (default: 2)
        Minimum number of nodes used during training.
    min_evaluate_nodes : int (default: 2)
        Minimum number of nodes used during validation.
    min_available_nodes : int (default: 2)
        Minimum number of total nodes in the system.
    num_malicious_nodes : int (default: 0)
        Number of malicious nodes in the system.
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
    selection_rule : Optional[Callable] (default: None)
        Function with signature (list[RecordDict], int, int) -> list[RecordDict].
        The inputs are:
        - a list of contents from reply messages,
        - the assumed number of malicious nodes (`num_malicious_nodes`),
        - the number of nodes to select (`num_nodes_to_select`).

        The function should implement a Byzantine-resilient selection rule that
        serves as the first step of Bulyan. If None, defaults to `select_multikrum`,
        which selects nodes according to the Multi-Krum algorithm.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        num_malicious_nodes: int = 0,
        weighted_by_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
        train_metrics_aggr_fn: (
            Callable[[list[RecordDict], str], MetricRecord] | None
        ) = None,
        evaluate_metrics_aggr_fn: (
            Callable[[list[RecordDict], str], MetricRecord] | None
        ) = None,
        selection_rule: (
            Callable[[list[RecordDict], int, int], list[RecordDict]] | None
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
        self.num_malicious_nodes = num_malicious_nodes
        self.selection_rule = selection_rule or select_multikrum

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> Bulyan settings:")
        log(INFO, "\t│\t├── Number of malicious nodes: %d", self.num_malicious_nodes)
        log(INFO, "\t│\t└── Selection rule: %s", self.selection_rule.__name__)
        super().summary()

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        # Check if sufficient replies have been received
        if len(valid_replies) < 4 * self.num_malicious_nodes + 3:
            log(
                WARN,
                "Insufficient replies, skipping Bulyan aggregation: "
                "Required at least %d (4*num_malicious_nodes + 3), but received %d.",
                4 * self.num_malicious_nodes + 3,
                len(valid_replies),
            )
            return None, None

        reply_contents = [msg.content for msg in valid_replies]

        # Compute theta and beta
        theta = len(valid_replies) - 2 * self.num_malicious_nodes
        beta = theta - 2 * self.num_malicious_nodes

        # Byzantine-resilient selection rule
        selected_contents = self.selection_rule(
            reply_contents, self.num_malicious_nodes, theta
        )

        # Convert each ArrayRecord to a list of NDArray for easier computation
        key = list(selected_contents[0].array_records.keys())[0]
        array_keys = list(selected_contents[0][key].keys())
        selected_ndarrays = [
            cast(ArrayRecord, ctnt[key]).to_numpy_ndarrays(keep_input=False)
            for ctnt in selected_contents
        ]

        # Compute median
        median_ndarrays = [
            np.median(arr, axis=0) for arr in zip(*selected_ndarrays, strict=True)
        ]

        # Aggregate the beta closest weights element-wise
        aggregated_ndarrays = aggregate_n_closest_weights(
            median_ndarrays, selected_ndarrays, beta
        )

        # Convert to ArrayRecord
        arrays = ArrayRecord(
            dict(
                zip(
                    array_keys,
                    (Array(np.asarray(arr)) for arr in aggregated_ndarrays),
                    strict=True,
                )
            )
        )

        # Aggregate MetricRecords
        metrics = self.train_metrics_aggr_fn(
            selected_contents,
            self.weighted_by_key,
        )
        return arrays, metrics


def aggregate_n_closest_weights(
    ref_weights: NDArrays, weights_list: list[NDArrays], beta: int
) -> NDArrays:
    """Compute the element-wise mean of the `beta` closest weight arrays.

    For each element (i-th coordinate), the output is the average of the
    `beta` weight arrays that are closest to the reference weights.

    Parameters
    ----------
    ref_weights : NDArrays
        Reference weights used to compute distances.
    weights_list : list[NDArrays]
        List of weight arrays (e.g., from selected nodes).
    beta : int
        Number of closest weight arrays to include in the averaging.

    Returns
    -------
    aggregated_weights : NDArrays
        Element-wise average of the `beta` closest weight arrays to the
        reference weights.
    """
    aggregated_weights = []
    for layer_id, ref_layer in enumerate(ref_weights):
        # Shape: (n_models, *layer_shape)
        layer_stack = np.stack([weights[layer_id] for weights in weights_list])

        # Compute absolute differences: shape (n_models, *layer_shape)
        diffs = np.abs(layer_stack - ref_layer)

        # Find indices of `beta` smallest per coordinate
        idx = np.argpartition(diffs, beta - 1, axis=0)[:beta]

        # Gather the closest weights
        closest = np.take_along_axis(layer_stack, idx, axis=0)

        # Average them
        aggregated_weights.append(np.mean(closest, axis=0))

    return aggregated_weights
