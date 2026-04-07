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
"""Federated Averaging with Trimmed Mean [Dong Yin, et al., 2021].

Paper: arxiv.org/abs/1803.01498
"""


from collections.abc import Callable, Iterable
from logging import INFO
from typing import cast

import numpy as np

from flwr.common import Array, ArrayRecord, Message, MetricRecord, NDArray, RecordDict
from flwr.common.logger import log

from ..exception import AggregationError
from .fedavg import FedAvg


class FedTrimmedAvg(FedAvg):
    """Federated Averaging with Trimmed Mean [Dong Yin, et al., 2021].

    Implemented based on: https://arxiv.org/abs/1803.01498

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
    beta : float (default: 0.2)
        Fraction to cut off of both tails of the distribution.
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
        beta: float = 0.2,
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
        self.beta = beta

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> FedTrimmedAvg settings:")
        log(INFO, "\t│\t└── beta: %s", self.beta)
        super().summary()

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

        # Aggregate ArrayRecords using trimmed mean
        # Get the key for the only ArrayRecord from the first Message
        record_key = list(valid_replies[0].content.array_records.keys())[0]
        # Preserve keys for arrays in ArrayRecord
        array_keys = list(valid_replies[0].content[record_key].keys())

        # Compute trimmed mean for each layer and construct ArrayRecord
        arrays = ArrayRecord()
        for array_key in array_keys:
            # Get the corresponding layer from each client
            layers = [
                cast(ArrayRecord, msg.content[record_key]).pop(array_key).numpy()
                for msg in valid_replies
            ]
            # Compute trimmed mean and save as Array in ArrayRecord
            try:
                arrays[array_key] = Array(
                    np.asarray(trim_mean(np.stack(layers), self.beta))
                )
            except ValueError as e:
                raise AggregationError(
                    f"Trimmed mean could not be computed. "
                    f"Likely cause: beta={self.beta} is too large."
                ) from e

        # Aggregate MetricRecords
        metrics = self.train_metrics_aggr_fn(
            [msg.content for msg in valid_replies],
            self.weighted_by_key,
        )
        return arrays, metrics


def trim_mean(array: NDArray, cut_fraction: float) -> NDArray:
    """Compute trimmed mean along axis=0.

    It is based on the scipy implementation:

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.trim_mean.html
    """
    axis = 0
    nobs = array.shape[0]
    lowercut = int(cut_fraction * nobs)
    uppercut = nobs - lowercut
    if lowercut > uppercut:
        raise ValueError("Fraction too big.")

    atmp = np.partition(array, (lowercut, uppercut - 1), axis)

    slice_list = [slice(None)] * atmp.ndim
    slice_list[axis] = slice(lowercut, uppercut)
    result: NDArray = np.mean(atmp[tuple(slice_list)], axis=axis)
    return result
