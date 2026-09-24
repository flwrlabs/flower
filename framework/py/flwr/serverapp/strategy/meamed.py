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
"""Mean-around-Median (MeaMed) [Xie et al., 2018] strategy.

Paper: arxiv.org/abs/1802.10116
"""


from collections.abc import Callable, Iterable
from logging import INFO
from typing import cast

import numpy as np

from flwr.common import Array, ArrayRecord, Message, MetricRecord, RecordDict
from flwr.common.logger import log

from .fedavg import FedAvg


class MeaMed(FedAvg):
    """Mean-around-Median (MeaMed) strategy.

    Implementation based on https://arxiv.org/abs/1802.10116

    MeaMed is a Byzantine-resilient aggregation strategy. It first computes
    the coordinate-wise median, then averages the ``num_closest`` values closest
    to that median for each coordinate.

    Parameters
    ----------
    fraction_train : float (default: 1.0)
        Fraction of nodes used during training. In case ``min_train_nodes``
        is larger than ``fraction_train * total_connected_nodes``,
        ``min_train_nodes`` will still be sampled.
    fraction_evaluate : float (default: 1.0)
        Fraction of nodes used during validation. In case ``min_evaluate_nodes``
        is larger than ``fraction_evaluate * total_connected_nodes``,
        ``min_evaluate_nodes`` will still be sampled.
    min_train_nodes : int (default: 2)
        Minimum number of nodes used during training.
    min_evaluate_nodes : int (default: 2)
        Minimum number of nodes used during validation.
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
        If ``None``, defaults to ``aggregate_metricrecords``, which performs a
        weighted average using the provided weight factor key.
    evaluate_metrics_aggr_fn : Optional[callable] (default: None)
        Function with signature (list[RecordDict], str) -> MetricRecord,
        used to aggregate MetricRecords from training round replies.
        If ``None``, defaults to ``aggregate_metricrecords``, which performs a
        weighted average using the provided weight factor key.
    num_closest : int (default: 2)
        Number of closest values to the median to average per coordinate.
    """

    def __init__(  # pylint: disable=R0913, R0917
        self,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
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
        num_closest: int = 2,
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
        self.num_closest = num_closest

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> MeaMed settings:")
        log(INFO, "\t│\t└── num_closest: %s", self.num_closest)
        super().summary()

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        if not valid_replies:
            return None, None

        # Get the key for the only ArrayRecord from the first Message
        record_key = list(valid_replies[0].content.array_records.keys())[0]
        # Preserve keys for arrays in ArrayRecord
        array_keys = list(valid_replies[0].content[record_key].keys())

        # Compute mean-around-median for each layer
        arrays = ArrayRecord()
        for array_key in array_keys:
            # Get the corresponding layer from each client
            layers = [
                cast(ArrayRecord, msg.content[record_key]).pop(array_key).numpy()
                for msg in valid_replies
            ]
            stacked = np.stack(layers)

            # Step 1: Compute coordinate-wise median
            median = np.median(stacked, axis=0)

            # Step 2: Find the num_closest values to the median per coordinate
            diff = np.abs(stacked - median)
            indices = np.argpartition(
                diff, kth=self.num_closest - 1, axis=0
            )[: self.num_closest]

            # Step 3: Average the closest values
            closest = np.take_along_axis(stacked, indices, axis=0)
            arrays[array_key] = Array(np.asarray(np.mean(closest, axis=0)))

        # Aggregate MetricRecords
        metrics = self.train_metrics_aggr_fn(
            [msg.content for msg in valid_replies],
            self.weighted_by_key,
        )
        return arrays, metrics
