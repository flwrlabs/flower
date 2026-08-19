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
from math import isfinite
from typing import cast

import numpy as np

from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log
from flwr.supercore.differential_privacy import (
    CLIENTS_DISCREPANCY_WARNING,
    KEY_CLIPPING_NORM,
    add_gaussian_noise_inplace,
    compute_clip_model_update,
    compute_stdv,
)
from flwr.supercore.privacy_accounting import (
    GaussianPrivacyEvent,
    PrivacyAccountant,
    PrivacySpent,
    SamplingMethod,
)

from ..exception import PrivacyBudgetExhausted
from ..grid import Grid
from .fedavg import FedAvg
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
    accountant : PrivacyAccountant or None
        Optional accountant used to track cumulative privacy loss. Accounted mode
        requires no-amplification accounting and uniform FedAvg aggregation.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        strategy: Strategy,
        noise_multiplier: float,
        clipping_norm: float,
        num_sampled_clients: int,
        *,
        accountant: PrivacyAccountant | None = None,
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

        if accountant is not None:
            if noise_multiplier <= 0:
                raise ValueError(
                    "The noise multiplier must be positive when accounting is enabled."
                )
            if accountant.config.sampling_method is not SamplingMethod.NO_AMPLIFICATION:
                raise ValueError(
                    "Fixed-clipping accounting currently requires "
                    "sampling_method='no-amplification'."
                )
            if not isinstance(strategy, FedAvg) or (
                strategy.__class__.aggregate_train is not FedAvg.aggregate_train
            ):
                raise ValueError(
                    "Fixed-clipping accounting currently requires a strategy using "
                    "FedAvg.aggregate_train."
                )

        self.noise_multiplier = noise_multiplier
        self.clipping_norm = clipping_norm
        self.num_sampled_clients = num_sampled_clients
        self.accountant = accountant
        self._pending_events: dict[int, GaussianPrivacyEvent | None] = {}

    def privacy_spent(self) -> PrivacySpent | None:
        """Return cumulative privacy expenditure, if accounting is enabled."""
        if self.accountant is None:
            return None
        return self.accountant.get_privacy_spent()

    def _prepare_accounting(
        self, server_round: int, messages: Iterable[Message]
    ) -> Iterable[Message]:
        """Create and check the privacy event for a configured round."""
        if self.accountant is None:
            return messages

        messages_list = list(messages)
        if server_round in self._pending_events:
            raise ValueError(
                f"Round {server_round} already has a pending privacy event."
            )
        if not messages_list:
            self._pending_events[server_round] = None
            return messages_list
        if len(messages_list) != self.num_sampled_clients:
            raise ValueError(
                "The configured sample size does not match num_sampled_clients: "
                f"{len(messages_list)} != {self.num_sampled_clients}."
            )

        event = GaussianPrivacyEvent(
            noise_multiplier=self.noise_multiplier,
            sample_size=len(messages_list),
            population_size=self.accountant.config.population_size,
        )
        if (
            self.accountant.config.max_epsilon is not None
            and self.accountant.would_exceed(event)
        ):
            spent = self.accountant.get_privacy_spent()
            raise PrivacyBudgetExhausted(
                "The next private release would exceed max_epsilon "
                f"(current epsilon: {spent.epsilon}, delta: {spent.delta})."
            )

        self._pending_events[server_round] = event
        return messages_list

    def _pop_pending_event(self, server_round: int) -> GaussianPrivacyEvent | None:
        """Take the pending event for aggregation of a configured round."""
        if self.accountant is None:
            return None
        if server_round not in self._pending_events:
            raise ValueError(f"Round {server_round} has no pending privacy event.")
        return self._pending_events.pop(server_round)

    def _validate_uniform_weights(self, replies: list[Message]) -> None:
        """Require equal client weights for the account-safe FedAvg path."""
        if self.accountant is None:
            return
        weighting_key = cast(FedAvg, self.strategy).weighted_by_key
        weights: list[float] = []
        for reply in replies:
            if len(reply.content.metric_records) != 1:
                raise ValueError(
                    "Accounted aggregation requires exactly one MetricRecord."
                )
            metrics = next(iter(reply.content.metric_records.values()))
            weight = metrics.get(weighting_key)
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not isfinite(weight)
                or weight <= 0
            ):
                raise ValueError(
                    "Accounted aggregation requires a positive finite client weight."
                )
            weights.append(float(weight))
        if any(weight != weights[0] for weight in weights[1:]):
            raise ValueError("Accounted aggregation requires equal client weights.")

    def _compose_release(self, event: GaussianPrivacyEvent | None) -> None:
        """Compose and report a successfully released private aggregate."""
        if self.accountant is None or event is None:
            return
        self.accountant.compose(event)
        spent = self.accountant.get_privacy_spent()
        log(
            INFO,
            "Privacy spent: epsilon=%.6f at delta=%g after %d releases",
            spent.epsilon,
            spent.delta,
            spent.num_releases,
        )

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
        if self.accountant is not None:
            spent = self.accountant.get_privacy_spent()
            log(
                INFO,
                "\t├──> Privacy spent: epsilon=%.6f at delta=%g",
                spent.epsilon,
                spent.delta,
            )
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
    accountant : PrivacyAccountant or None
        Optional accountant used to track cumulative privacy loss. Accounted mode
        requires no-amplification accounting and uniform FedAvg aggregation.

    Examples
    --------
    Create a strategy::

        strategy = fl.serverapp.FedAvg( ... )

    Wrap the strategy with the `DifferentialPrivacyServerSideFixedClipping` wrapper::

        dp_strategy = DifferentialPrivacyServerSideFixedClipping(
            strategy, cfg.noise_multiplier, cfg.clipping_norm, cfg.num_sampled_clients
        )
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        strategy: Strategy,
        noise_multiplier: float,
        clipping_norm: float,
        num_sampled_clients: int,
        *,
        accountant: PrivacyAccountant | None = None,
    ) -> None:
        super().__init__(
            strategy,
            noise_multiplier,
            clipping_norm,
            num_sampled_clients,
            accountant=accountant,
        )
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
        messages = self.strategy.configure_train(server_round, arrays, config, grid)
        return self._prepare_accounting(server_round, messages)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        replies_list = list(replies)
        event = self._pop_pending_event(server_round)
        if not validate_replies(
            replies_list,
            self.num_sampled_clients,
            strict=self.accountant is not None,
        ):
            return None, None
        self._validate_uniform_weights(replies_list)

        # Clip arrays in replies
        current_ndarrays = self.current_arrays.to_numpy_ndarrays()
        for reply in replies_list:
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
            server_round, replies_list
        )

        # Add Gaussian noise to the aggregated arrays
        if aggregated_arrays:
            aggregated_arrays = self._add_noise_to_aggregated_arrays(aggregated_arrays)
            self._compose_release(event)

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
    accountant : PrivacyAccountant or None
        Optional accountant used to track cumulative privacy loss. Accounted mode
        requires no-amplification accounting and uniform FedAvg aggregation.

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
        messages = self.strategy.configure_train(server_round, arrays, config, grid)
        return self._prepare_accounting(server_round, messages)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        replies_list = list(replies)
        event = self._pop_pending_event(server_round)
        if not validate_replies(
            replies_list,
            self.num_sampled_clients,
            strict=self.accountant is not None,
        ):
            return None, None
        self._validate_uniform_weights(replies_list)

        # Aggregate
        aggregated_arrays, aggregated_metrics = self.strategy.aggregate_train(
            server_round, replies_list
        )

        # Add Gaussian noise to the aggregated arrays
        if aggregated_arrays:
            aggregated_arrays = self._add_noise_to_aggregated_arrays(aggregated_arrays)
            self._compose_release(event)

        return aggregated_arrays, aggregated_metrics


def validate_replies(
    replies: Iterable[Message], num_sampled_clients: int, strict: bool = False
) -> bool:
    """Validate replies and log errors/warnings.

    Arguments
    ----------
    replies : Iterable[Message]
        The replies to validate.
    num_sampled_clients : int
        The expected number of sampled clients.
    strict : bool
        Whether a reply-count mismatch should abort aggregation.

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
        if strict:
            return False

    return True
