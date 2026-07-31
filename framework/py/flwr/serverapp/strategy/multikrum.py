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
"""Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent.

[Blanchard et al., 2017].

Paper: proceedings.neurips.cc/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf
"""


from collections.abc import Callable, Iterable
from logging import INFO
from typing import cast

import numpy as np

from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict
from flwr.common import NDArray, log

from .fedavg import FedAvg
from .strategy_utils import aggregate_arrayrecords


# pylint: disable=too-many-instance-attributes
class MultiKrum(FedAvg):
    """MultiKrum [Blanchard et al., 2017] strategy.

    Implementation based on https://arxiv.org/abs/1703.02757

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
        Number of malicious nodes in the system. Defaults to 0.
    num_nodes_to_select : int (default: 1)
        Number of nodes to select before averaging.
    weighted_by_key : str (default: "num-examples")
        The key within each MetricRecord whose value is used as the weight when
        computing weighted averages for both ArrayRecords and MetricRecords.
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

    Notes
    -----
    MultiKrum is a generalization of Krum. If `num_nodes_to_select` is set to 1,
    MultiKrum will reduce to classical Krum.
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
        num_nodes_to_select: int = 1,
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
        self.num_nodes_to_select = num_nodes_to_select

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> MultiKrum settings:")
        log(INFO, "\t│\t├── Number of malicious nodes: %d", self.num_malicious_nodes)
        log(INFO, "\t│\t└── Number of nodes to select: %d", self.num_nodes_to_select)
        super().summary()

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

            # Krum or MultiKrum selection
            replies_to_aggregate = select_multikrum(
                reply_contents,
                num_malicious_nodes=self.num_malicious_nodes,
                num_nodes_to_select=self.num_nodes_to_select,
            )

            # Aggregate ArrayRecords
            arrays = aggregate_arrayrecords(
                replies_to_aggregate,
                self.weighted_by_key,
            )

            # Aggregate MetricRecords
            metrics = self.train_metrics_aggr_fn(
                replies_to_aggregate,
                self.weighted_by_key,
            )
        return arrays, metrics


def compute_distances(records: list[ArrayRecord]) -> NDArray:
    """Compute squared L2 distances between ArrayRecords.

    Parameters
    ----------
    records : list[ArrayRecord]
        A list of ArrayRecords (arrays received in replies)

    Returns
    -------
    NDArray
        A 2D array representing the distance matrix of squared L2 distances
        between input ArrayRecords
    """
    # Formula: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x.y
    # Flatten records and stack them into a matrix
    flat_w = np.stack(
        [np.concatenate(rec.to_numpy_ndarrays(), axis=None).ravel() for rec in records],
        axis=0,
    )  # shape: (n, d) with n number of records and d the dimension of model

    # Compute squared norms of each vector
    norms: NDArray = np.square(flat_w).sum(axis=1)  # shape (n,)

    # Use broadcasting to compute pairwise distances
    distance_matrix: NDArray = norms[:, None] + norms[None, :] - 2 * flat_w @ flat_w.T
    return distance_matrix


def select_multikrum(
    contents: list[RecordDict],
    num_malicious_nodes: int,
    num_nodes_to_select: int,
) -> list[RecordDict]:
    """Select the set of RecordDicts to aggregate using the Krum or MultiKrum algorithm.

    For each node, computes the sum of squared L2 distances to its n-f-2 closest
    parameter vectors, where n is the number of nodes and f is the number of
    malicious nodes. The node(s) with the lowest score(s) are selected for
    aggregation.

    Parameters
    ----------
    contents : list[RecordDict]
        List of contents from reply messages, where each content is a RecordDict
        containing an ArrayRecord of model parameters from a node (client).
    num_malicious_nodes : int
        Number of malicious nodes in the system.
    num_nodes_to_select : int
        Number of client updates to select.
        - If 1, the algorithm reduces to classical Krum (selecting a single update).
        - If >1, Multi-Krum is applied (selecting multiple updates).

    Returns
    -------
    list[RecordDict]
        Selected contents following the Krum or Multi-Krum algorithm.

    Notes
    -----
    If `num_nodes_to_select` is set to 1, Multi-Krum reduces to classical Krum
    and only a single RecordDict is selected.
    """
    # Construct list of ArrayRecord objects from replies
    record_key = list(contents[0].array_records.keys())[0]
    # Recall aggregate_train first ensures replies only contain one ArrayRecord
    array_records = [cast(ArrayRecord, reply[record_key]) for reply in contents]
    distance_matrix = compute_distances(array_records)

    # For each node, take the n-f-2 closest parameters vectors
    num_closest = max(1, len(array_records) - num_malicious_nodes - 2)
    closest_indices = []
    for distance in distance_matrix:
        closest_indices.append(
            np.argsort(distance)[1 : num_closest + 1].tolist()  # noqa: E203
        )

    # Compute the score for each node, that is the sum of the distances
    # of the n-f-2 closest parameters vectors
    scores = [
        np.sum(distance_matrix[i, closest_indices[i]])
        for i in range(len(distance_matrix))
    ]

    # Choose the num_nodes_to_select lowest-scoring nodes (MultiKrum)
    # and return their updates
    best_indices = np.argsort(scores)[:num_nodes_to_select]
    return [contents[i] for i in best_indices]
