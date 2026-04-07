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
"""Federated Median (FedMedian) [Yin et al., 2018] strategy.

Paper: arxiv.org/pdf/1803.01498v1.pdf
"""


from collections.abc import Iterable
from typing import cast

import numpy as np

from flwr.common import Array, ArrayRecord, Message, MetricRecord

from .fedavg import FedAvg


class FedMedian(FedAvg):
    """Federated Median (FedMedian) strategy.

    Implementation based on https://arxiv.org/pdf/1803.01498v1

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

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        # Call FedAvg aggregate_train to perform validation and aggregation
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        if not valid_replies:
            return None, None

        # Aggregate ArrayRecords using median
        # Get the key for the only ArrayRecord from the first Message
        record_key = list(valid_replies[0].content.array_records.keys())[0]
        # Preserve keys for arrays in ArrayRecord
        array_keys = list(valid_replies[0].content[record_key].keys())

        # Compute median for each layer and construct ArrayRecord
        arrays = ArrayRecord()
        for array_key in array_keys:
            # Get the corresponding layer from each client
            layers = [
                cast(ArrayRecord, msg.content[record_key]).pop(array_key).numpy()
                for msg in valid_replies
            ]
            # Compute median and save as Array in ArrayRecord
            arrays[array_key] = Array(np.asarray(np.median(np.stack(layers), axis=0)))

        # Aggregate MetricRecords
        metrics = self.train_metrics_aggr_fn(
            [msg.content for msg in valid_replies],
            self.weighted_by_key,
        )
        return arrays, metrics
