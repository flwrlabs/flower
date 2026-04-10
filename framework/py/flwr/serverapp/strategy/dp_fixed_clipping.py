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
"""Message-based Central differential privacy with fixed clipping.

Papers: https://arxiv.org/abs/1712.07557, https://arxiv.org/abs/1710.06963
"""


from abc import ABC
from collections.abc import Iterable
from logging import INFO, WARNING

import numpy as np

from flwr.common import Array, ArrayRecord, ConfigRecord, Message, MetricRecord, log
from flwr.common.differential_privacy import (
    add_gaussian_noise_inplace,
    compute_clip_model_update,
    compute_stdv,
)
from flwr.common.differential_privacy_constants import (
    CLIENTS_DISCREPANCY_WARNING,
    KEY_CLIPPING_NORM,
)
from flwr.server import Grid

from .strategy import Strategy


class DifferentialPrivacyFixedClippingBase(Strategy, ABC):
    """Base class for DP strategies with fixed clipping.

    This class contains common functionality shared between server-side and
    client-side fixed clipping implementations.

    Parameters
    ----------
    strategy : Strategy
        The strategy to which DP functionalities will be added by this wrapper.
    noise_multiplier : float
        The noise multiplier for the Gaussian mechanism for model updates.
        A value of 1.0 or higher is recommended for strong privacy.
    clipping_norm : float
        The value of the clipping norm.
    num_sampled_clients : int
        The number of clients that are sampled on each round.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        strategy: Strategy,
        noise_multiplier: float,
        clipping_norm: float,
        num_sampled_clients: int,
    ) -> None:
        super().__init__()

        self.strategy = strategy

        if noise_multiplier < 0:
            raise ValueError("The noise multiplier should be a non-negative value.")

        if clipping_norm <= 0:
            raise ValueError("The clipping norm should be a positive value.")

        if num_sampled_clients <= 0:
            raise ValueError(
                "The number of sampled clients should be a positive value."
            )

        self.noise_multiplier = noise_multiplier
        self.clipping_norm = clipping_norm
        self.num_sampled_clients = num_sampled_clients

    def _add_noise_to_aggregated_arrays(
        self, aggregated_arrays: ArrayRecord
    ) -> ArrayRecord:
        """Add Gaussian noise to aggregated arrays.

        Parameters
        ----------
        aggregated_arrays : ArrayRecord
            The aggregated arrays to add noise to.

        Returns
        -------
        ArrayRecord
            The aggregated arrays with noise added.
        """
        aggregated_ndarrays = aggregated_arrays.to_numpy_ndarrays()
        stdv = compute_stdv(
            self.noise_multiplier, self.clipping_norm, self.num_sampled_clients
        )
        add_gaussian_noise_inplace(aggregated_ndarrays, stdv)

        log(
            INFO,
            "aggregate_fit: central DP noise with %.4f stdev added",
            stdv,
        )

        return ArrayRecord(
            {
                k: Array(np.asarray(v))
                for k, v in zip(
                    aggregated_arrays.keys(), aggregated_ndarrays, strict=True
                )
            }
        )

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated evaluation."""
        return self.strategy.configure_evaluate(server_round, arrays, config, grid)

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> MetricRecord | None:
        """Aggregate MetricRecords in the received Messages."""
        return self.strategy.aggregate_evaluate(server_round, replies)

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        self.strategy.summary()


class DifferentialPrivacyServerSideFixedClipping(DifferentialPrivacyFixedClippingBase):
    """Strategy wrapper for central DP with server-side fixed clipping.

    Parameters
    ----------
    strategy : Strategy
        The strategy to which DP functionalities will be added by this wrapper.
    noise_multiplier : float
        The noise multiplier for the Gaussian mechanism for model updates.
        A value of 1.0 or higher is recommended for strong privacy.
    clipping_norm : float
        The value of the clipping norm.
    num_sampled_clients : int
        The number of clients that are sampled on each round.

    Examples
    --------
    Create a strategy::

        strategy = fl.serverapp.FedAvg( ... )

    Wrap the strategy with the `DifferentialPrivacyServerSideFixedClipping` wrapper::

        dp_strategy = DifferentialPrivacyServerSideFixedClipping(
            strategy, cfg.noise_multiplier, cfg.clipping_norm, cfg.num_sampled_clients
        )
    """

    def __init__(
        self,
        strategy: Strategy,
        noise_multiplier: float,
        clipping_norm: float,
        num_sampled_clients: int,
    ) -> None:
        super().__init__(strategy, noise_multiplier, clipping_norm, num_sampled_clients)
        self.current_arrays: ArrayRecord = ArrayRecord()

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        return "Differential Privacy Strategy Wrapper (Server-Side Fixed Clipping)"

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> DP settings:")
        log(INFO, "\t│\t├── Noise multiplier: %s", self.noise_multiplier)
        log(INFO, "\t│\t└── Clipping norm: %s", self.clipping_norm)
        super().summary()

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of training."""
        self.current_arrays = arrays
        return self.strategy.configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        if not validate_replies(replies, self.num_sampled_clients):
            return None, None

        # Clip arrays in replies
        current_ndarrays = self.current_arrays.to_numpy_ndarrays()
        for reply in replies:
            for arr_name, record in reply.content.array_records.items():
                # Clip
                reply_ndarrays = record.to_numpy_ndarrays()
                compute_clip_model_update(
                    param1=reply_ndarrays,
                    param2=current_ndarrays,
                    clipping_norm=self.clipping_norm,
                )
                # Replace content while preserving keys
                reply.content[arr_name] = ArrayRecord(
                    dict(
                        zip(
                            record.keys(),
                            (Array(np.asarray(v)) for v in reply_ndarrays),
                            strict=True,
                        )
                    )
                )
            log(
                INFO,
                "aggregate_fit: parameters are clipped by value: %.4f.",
                self.clipping_norm,
            )

        # Pass the new parameters for aggregation
        aggregated_arrays, aggregated_metrics = self.strategy.aggregate_train(
            server_round, replies
        )

        # Add Gaussian noise to the aggregated arrays
        if aggregated_arrays:
            aggregated_arrays = self._add_noise_to_aggregated_arrays(aggregated_arrays)

        return aggregated_arrays, aggregated_metrics


class DifferentialPrivacyClientSideFixedClipping(DifferentialPrivacyFixedClippingBase):
    """Strategy wrapper for central DP with client-side fixed clipping.

    Use `fixedclipping_mod` modifier at the client side.

    In comparison to `DifferentialPrivacyServerSideFixedClipping`,
    which performs clipping on the server-side,
    `DifferentialPrivacyClientSideFixedClipping` expects clipping to happen
    on the client-side, usually by using the built-in `fixedclipping_mod`.

    Parameters
    ----------
    strategy : Strategy
        The strategy to which DP functionalities will be added by this wrapper.
    noise_multiplier : float
        The noise multiplier for the Gaussian mechanism for model updates.
        A value of 1.0 or higher is recommended for strong privacy.
    clipping_norm : float
        The value of the clipping norm.
    num_sampled_clients : int
        The number of clients that are sampled on each round.

    Examples
    --------
    Create a strategy::

        strategy = fl.serverapp.FedAvg(...)

    Wrap the strategy with the `DifferentialPrivacyClientSideFixedClipping` wrapper::

        dp_strategy = DifferentialPrivacyClientSideFixedClipping(
            strategy, cfg.noise_multiplier, cfg.clipping_norm, cfg.num_sampled_clients
        )

    On the client, add the `fixedclipping_mod` to the client-side mods::

        app = fl.client.ClientApp(mods=[fixedclipping_mod])
    """

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        return "Differential Privacy Strategy Wrapper (Client-Side Fixed Clipping)"

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> DP settings:")
        log(INFO, "\t│\t├── Noise multiplier: %s", self.noise_multiplier)
        log(INFO, "\t│\t└── Clipping norm: %s", self.clipping_norm)
        super().summary()

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of training."""
        # Inject clipping norm in config
        config[KEY_CLIPPING_NORM] = self.clipping_norm
        # Call parent method
        return self.strategy.configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        if not validate_replies(replies, self.num_sampled_clients):
            return None, None

        # Aggregate
        aggregated_arrays, aggregated_metrics = self.strategy.aggregate_train(
            server_round, replies
        )

        # Add Gaussian noise to the aggregated arrays
        if aggregated_arrays:
            aggregated_arrays = self._add_noise_to_aggregated_arrays(aggregated_arrays)

        return aggregated_arrays, aggregated_metrics


def validate_replies(replies: Iterable[Message], num_sampled_clients: int) -> bool:
    """Validate replies and log errors/warnings.

    Arguments
    ----------
    replies : Iterable[Message]
        The replies to validate.
    num_sampled_clients : int
        The expected number of sampled clients.

    Returns
    -------
    bool
        True if replies are valid for aggregation, False otherwise.
    """
    num_errors = 0
    num_replies_with_content = 0
    for msg in replies:
        if msg.has_error():
            log(
                INFO,
                "Received error in reply from node %d: %s",
                msg.metadata.src_node_id,
                msg.error,
            )
            num_errors += 1
        else:
            num_replies_with_content += 1

    # Errors are not allowed
    if num_errors:
        log(
            INFO,
            "aggregate_train: Some clients reported errors. Skipping aggregation.",
        )
        return False

    log(
        INFO,
        "aggregate_train: Received %s results and %s failures",
        num_replies_with_content,
        num_errors,
    )

    if num_replies_with_content != num_sampled_clients:
        log(
            WARNING,
            CLIENTS_DISCREPANCY_WARNING,
            num_replies_with_content,
            num_sampled_clients,
        )

    return True
