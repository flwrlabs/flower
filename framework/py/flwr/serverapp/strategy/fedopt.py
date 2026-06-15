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
"""Adaptive Federated Optimization (FedOpt) [Reddi et al., 2020] abstract strategy.

Paper: arxiv.org/abs/2003.00295
"""

from collections.abc import Callable, Iterable
from logging import INFO

import numpy as np

from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.common import NDArray, log

from ..exception import AggregationError
from ..grid import Grid
from .fedavg import FedAvg


# pylint: disable=line-too-long
class FedOpt(FedAvg):
    """Federated Optim strategy.

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
    beta_1 : float, optional
        Momentum parameter. Defaults to 0.0.
    beta_2 : float, optional
        Second moment parameter. Defaults to 0.0.
    tau : float, optional
        Controls the algorithm's degree of adaptability. Defaults to 1e-3.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-locals, line-too-long
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
        beta_1: float = 0.0,
        beta_2: float = 0.0,
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
        )
        self.current_arrays: dict[str, NDArray] | None = None
        self.eta = eta
        self.eta_l = eta_l
        self.tau = tau
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m_t: dict[str, NDArray] | None = None
        self.v_t: dict[str, NDArray] | None = None

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> FedOpt settings:")
        log(
            INFO,
            "\t│\t├── eta (%s) | eta_l (%s)",
            f"{self.eta:.6g}",
            f"{self.eta_l:.6g}",
        )
        log(
            INFO,
            "\t│\t├── beta_1 (%s) | beta_2 (%s)",
            f"{self.beta_1:.6g}",
            f"{self.beta_2:.6g}",
        )
        log(
            INFO,
            "\t│\t└── tau (%s)",
            f"{self.tau:.6g}",
        )
        super().summary()

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        # Keep track of array record being communicated
        self.current_arrays = {k: array.numpy() for k, array in arrays.items()}
        return super().configure_train(server_round, arrays, config, grid)

    def _compute_deltat_and_mt(
        self, aggregated_arrayrecord: ArrayRecord
    ) -> tuple[dict[str, NDArray], dict[str, NDArray], dict[str, NDArray]]:
        """Compute delta_t and m_t.

        This is a shared stage during aggregation for FedAdagrad, FedAdam and FedYogi.
        """
        if self.current_arrays is None:
            reason = (
                "Current arrays not set. Ensure that `configure_train` has been "
                "called before aggregation."
            )
            raise AggregationError(reason=reason)

        aggregated_ndarrays = {
            k: array.numpy() for k, array in aggregated_arrayrecord.items()
        }

        # Check keys in aggregated arrays match those in current arrays
        if set(aggregated_ndarrays.keys()) != set(self.current_arrays.keys()):
            reason = (
                "Keys of the aggregated arrays do not match those of the arrays "
                "stored at the strategy. `delta_t = aggregated_arrays - "
                "current_arrays` cannot be computed."
            )
            raise AggregationError(reason=reason)

        # Check that the shape of values match
        # Only shapes that match can compute delta_t (we don't want
        # broadcasting to happen)
        for k, x in aggregated_ndarrays.items():
            if x.shape != self.current_arrays[k].shape:
                reason = (
                    f"Shape of aggregated array '{k}' does not match "
                    f"shape of the array under the same key stored in the strategy. "
                    f"Cannot compute `delta_t`."
                )
                raise AggregationError(reason=reason)

        delta_t = {
            k: x - self.current_arrays[k] for k, x in aggregated_ndarrays.items()
        }

        # m_t
        if not self.m_t:
            self.m_t = {k: np.zeros_like(v) for k, v in aggregated_ndarrays.items()}
        self.m_t = {
            k: self.beta_1 * v + (1 - self.beta_1) * delta_t[k]
            for k, v in self.m_t.items()
        }

        return delta_t, self.m_t, aggregated_ndarrays
