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
"""Federated Optimization (FedProx) [Li et al., 2018] strategy.

Paper: arxiv.org/abs/1812.06127
"""


from collections.abc import Callable, Iterable
from logging import INFO, WARN

from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.common import log

from ..grid import Grid
from .fedavg import FedAvg


class FedProx(FedAvg):
    r"""Federated Optimization strategy.

    Implementation based on https://arxiv.org/abs/1812.06127

    FedProx extends FedAvg by introducing a proximal term into the client-side
    optimization objective. The strategy itself behaves identically to FedAvg
    on the server side, but each client **MUST** add a proximal regularization
    term to its local loss function during training:

    .. math::
        \frac{\mu}{2} || w - w^t ||^2

    Where $w^t$ denotes the global parameters and $w$ denotes the local weights
    being optimized.

    This strategy sends the proximal term inside the ``ConfigRecord`` as part of the
    ``configure_train`` method under key ``"proximal-mu"``. The client can then use this
    value to add the proximal term to the loss function.

    In PyTorch, for example, the loss would go from:

    .. code:: python
        loss = criterion(net(inputs), labels)

    To:

    .. code:: python
        # Get proximal term weight from message
        mu = msg.content["config"]["proximal-mu"]

        # Compute proximal term
        proximal_term = 0.0
        for local_weights, global_weights in zip(net.parameters(), global_params):
            proximal_term += (local_weights - global_weights).norm(2)

        # Update loss
        loss = criterion(net(inputs), labels) + (mu / 2) * proximal_term

    With ``global_params`` being a copy of the model parameters, created **after**
    applying the received global weights but **before** local training begins.

    .. code:: python
        global_params = copy.deepcopy(net).parameters()

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
    proximal_mu : float (default: 0.0)
        The weight of the proximal term used in the optimization. 0.0 makes
        this strategy equivalent to FedAvg, and the higher the coefficient, the more
        regularization will be used (that is, the client parameters will need to be
        closer to the server parameters during training).
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
        proximal_mu: float = 0.0,
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
        self.proximal_mu = proximal_mu

        if self.proximal_mu == 0.0:
            log(
                WARN,
                "FedProx initialized with `proximal_mu=0.0`. "
                "This makes the strategy equivalent to FedAvg.",
            )

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> FedProx settings:")
        log(INFO, "\t│\t└── Proximal mu: %s", self.proximal_mu)
        super().summary()

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        # Inject proximal term weight into config
        config["proximal-mu"] = self.proximal_mu
        return super().configure_train(server_round, arrays, config, grid)
