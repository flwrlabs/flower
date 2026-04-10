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
"""FedAdagrad [Reddi et al., 2020] strategy.

Adaptive Federated Optimization using Adagrad.

Paper: arxiv.org/abs/2003.00295
"""


from collections.abc import Callable, Iterable

import numpy as np

from flwr.common import Array, ArrayRecord, Message, MetricRecord, RecordDict

from ..exception import AggregationError
from .fedopt import FedOpt


# pylint: disable=line-too-long
class FedAdagrad(FedOpt):
    """FedAdagrad strategy - Adaptive Federated Optimization using Adagrad.

    Implementation based on https://arxiv.org/abs/2003.00295v5

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
    eta : float, optional
        Server-side learning rate. Defaults to 1e-1.
    eta_l : float, optional
        Client-side learning rate. Defaults to 1e-1.
    tau : float, optional
        Controls the algorithm's degree of adaptability. Defaults to 1e-3.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        *,
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
        eta: float = 1e-1,
        eta_l: float = 1e-1,
        tau: float = 1e-3,
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
            eta=eta,
            eta_l=eta_l,
            beta_1=0.0,
            beta_2=0.0,
            tau=tau,
        )

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        aggregated_arrayrecord, aggregated_metrics = super().aggregate_train(
            server_round, replies
        )

        if aggregated_arrayrecord is None:
            return aggregated_arrayrecord, aggregated_metrics

        if self.current_arrays is None:
            reason = (
                "Current arrays not set. Ensure that `configure_train` has been "
                "called before aggregation."
            )
            raise AggregationError(reason=reason)

        # Compute intermediate variables
        delta_t, m_t, aggregated_ndarrays = self._compute_deltat_and_mt(
            aggregated_arrayrecord
        )

        # v_t
        if not self.v_t:
            self.v_t = {k: np.zeros_like(v) for k, v in aggregated_ndarrays.items()}
        self.v_t = {k: v + (delta_t[k] ** 2) for k, v in self.v_t.items()}

        new_arrays = {
            k: x + self.eta * m_t[k] / (np.sqrt(self.v_t[k]) + self.tau)
            for k, x in self.current_arrays.items()
        }

        return (
            ArrayRecord({k: Array(np.asarray(v)) for k, v in new_arrays.items()}),
            aggregated_metrics,
        )
