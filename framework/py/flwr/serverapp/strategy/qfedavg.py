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
"""Fair Resource Allocation in Federated Learning [Li et al., 2020] strategy.

Paper: openreview.net/pdf?id=ByexElSYDr
"""


from collections.abc import Callable, Iterable
from logging import INFO

import numpy as np

from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.common import NDArray
from flwr.common.logger import log

from ..exception import AggregationError
from ..grid import Grid
from .fedavg import FedAvg


class QFedAvg(FedAvg):
    """Q-FedAvg strategy.

    Implementation based on openreview.net/pdf?id=ByexElSYDr

    Parameters
    ----------
    client_learning_rate : float
        Local learning rate used by clients during training. This value is used by
        the strategy to approximate the base Lipschitz constant L, via
        L = 1 / client_learning_rate.
    q : float (default: 0.1)
        The parameter q that controls the degree of fairness of the algorithm. Please
        tune this parameter based on your use case.
        When set to 0, q-FedAvg is equivalent to FedAvg.
    train_loss_key : str (default: "train_loss")
        The key within the MetricRecord whose value is used as the training loss when
        aggregating ArrayRecords following q-FedAvg.
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

    def __init__(  # pylint: disable=R0913, R0917
        self,
        client_learning_rate: float,
        q: float = 0.1,
        train_loss_key: str = "train_loss",
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
        self.q = q
        self.client_learning_rate = client_learning_rate
        self.train_loss_key = train_loss_key
        self.current_arrays: ArrayRecord | None = None

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> q-FedAvg settings:")
        log(INFO, "\t│\t├── client_learning_rate: %s", self.client_learning_rate)
        log(INFO, "\t│\t├── q: %s", self.q)
        log(INFO, "\t│\t└── train_loss_key: '%s'", self.train_loss_key)
        super().summary()

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        self.current_arrays = arrays.copy()
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(  # pylint: disable=too-many-locals
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        # Call FedAvg aggregate_train to perform validation and aggregation
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        if not valid_replies:
            return None, None

        # Compute estimate of Lipschitz constant L
        L = 1.0 / self.client_learning_rate  # pylint: disable=C0103

        # q-FedAvg aggregation
        if self.current_arrays is None:
            raise AggregationError(
                "Current global model weights are not available. Make sure to call"
                "`configure_train` before calling `aggregate_train`."
            )
        array_keys = list(self.current_arrays.keys())  # Preserve keys
        global_weights = self.current_arrays.to_numpy_ndarrays(keep_input=False)
        sum_delta = None
        sum_h = 0.0

        for msg in valid_replies:
            # Extract local weights and training loss from Message
            local_weights = get_local_weights(msg)
            loss = get_train_loss(msg, self.train_loss_key)

            # Compute delta and h
            delta, h = compute_delta_and_h(
                global_weights, local_weights, self.q, L, loss
            )

            # Compute sum of deltas and sum of h
            if sum_delta is None:
                sum_delta = delta
            else:
                sum_delta = [sd + d for sd, d in zip(sum_delta, delta, strict=True)]
            sum_h += h

        # Compute new global weights and convert to Array type
        # `np.asarray` can convert numpy scalars to 0-dim arrays
        assert sum_delta is not None  # Make mypy happy
        array_list = [
            Array(np.asarray(gw - (d / sum_h)))
            for gw, d in zip(global_weights, sum_delta, strict=True)
        ]

        # Aggregate MetricRecords
        metrics = self.train_metrics_aggr_fn(
            [msg.content for msg in valid_replies],
            self.weighted_by_key,
        )
        return (
            ArrayRecord(dict(zip(array_keys, array_list, strict=True))),
            metrics,
        )


def get_train_loss(msg: Message, loss_key: str) -> float:
    """Extract training loss from a Message."""
    metrics = list(msg.content.metric_records.values())[0]
    if (loss := metrics.get(loss_key)) is None or not isinstance(loss, (int | float)):
        raise AggregationError(
            "Missing or invalid training loss. "
            f"The strategy expected a float value for the key '{loss_key}' "
            "as the training loss in each MetricRecord from the clients. "
            f"Ensure that '{loss_key}' is present and maps to a valid float."
        )
    return float(loss)


def get_local_weights(msg: Message) -> list[NDArray]:
    """Extract local weights from a Message."""
    arrays = list(msg.content.array_records.values())[0]
    return arrays.to_numpy_ndarrays(keep_input=False)


def l2_norm(ndarrays: list[NDArray]) -> float:
    """Compute the squared L2 norm of a list of numpy.ndarray."""
    return float(sum(np.sum(np.square(g)) for g in ndarrays))


def compute_delta_and_h(
    global_weights: list[NDArray],
    local_weights: list[NDArray],
    q: float,
    L: float,  # Lipschitz constant  # pylint: disable=C0103
    loss: float,
) -> tuple[list[NDArray], float]:
    """Compute delta and h used in q-FedAvg aggregation."""
    # Compute gradient_k = L * (w - w_k)
    for gw, lw in zip(global_weights, local_weights, strict=True):
        np.subtract(gw, lw, out=lw)
        lw *= L
    grad = local_weights  # After in-place operations, local_weights is now grad
    # Compute ||w_k - w||^2
    norm = l2_norm(grad)
    # Compute delta_k = loss_k^q * gradient_k
    loss_pow_q: float = np.float_power(loss + 1e-10, q)
    for g in grad:
        g *= loss_pow_q
    delta = grad  # After in-place multiplication, grad is now delta
    # Compute h_k
    h = q * np.float_power(loss + 1e-10, q - 1) * norm + L * loss_pow_q
    return delta, h
