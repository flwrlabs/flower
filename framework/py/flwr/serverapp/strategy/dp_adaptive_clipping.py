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
"""Message-based Central differential privacy with adaptive clipping.

Paper (Andrew et al.): https://arxiv.org/abs/1905.03871
"""

import math
from abc import ABC
from collections.abc import Iterable
from logging import INFO

import numpy as np

from flwr.common import Array, ArrayRecord, ConfigRecord, Message, MetricRecord, log
from flwr.common.differential_privacy import (
    adaptive_clip_inputs_inplace,
    add_gaussian_noise_inplace,
    compute_adaptive_noise_params,
    compute_stdv,
)
from flwr.common.differential_privacy_constants import KEY_CLIPPING_NORM, KEY_NORM_BIT
from flwr.server import Grid
from flwr.serverapp.exception import AggregationError

from .dp_fixed_clipping import validate_replies
from .strategy import Strategy


class DifferentialPrivacyAdaptiveBase(Strategy, ABC):
    """Base class for DP strategies with adaptive clipping."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-positional-arguments
    def __init__(
        self,
        strategy: Strategy,
        noise_multiplier: float,
        num_sampled_clients: int,
        initial_clipping_norm: float = 0.1,
        target_clipped_quantile: float = 0.5,
        clip_norm_lr: float = 0.2,
        clipped_count_stddev: float | None = None,
    ) -> None:
        super().__init__()

        if strategy is None:
            raise ValueError("The passed strategy is None.")
        if noise_multiplier < 0:
            raise ValueError("The noise multiplier should be a non-negative value.")
        if num_sampled_clients <= 0:
            raise ValueError(
                "The number of sampled clients should be a positive value."
            )
        if initial_clipping_norm <= 0:
            raise ValueError("The initial clipping norm should be a positive value.")
        if not 0 <= target_clipped_quantile <= 1:
            raise ValueError("The target clipped quantile must be in [0, 1].")
        if clip_norm_lr <= 0:
            raise ValueError("The learning rate must be positive.")
        if clipped_count_stddev is not None and clipped_count_stddev < 0:
            raise ValueError("The `clipped_count_stddev` must be non-negative.")

        self.strategy = strategy
        self.num_sampled_clients = num_sampled_clients
        self.clipping_norm = initial_clipping_norm
        self.target_clipped_quantile = target_clipped_quantile
        self.clip_norm_lr = clip_norm_lr
        (
            self.clipped_count_stddev,
            self.noise_multiplier,
        ) = compute_adaptive_noise_params(
            noise_multiplier,
            num_sampled_clients,
            clipped_count_stddev,
        )

    def _add_noise_to_aggregated_arrays(self, aggregated: ArrayRecord) -> ArrayRecord:
        nds = aggregated.to_numpy_ndarrays()
        stdv = compute_stdv(
            self.noise_multiplier, self.clipping_norm, self.num_sampled_clients
        )
        add_gaussian_noise_inplace(nds, stdv)
        log(INFO, "aggregate_fit: central DP noise with %.4f stdev added", stdv)
        return ArrayRecord(
            {
                k: Array(np.asarray(v))
                for k, v in zip(aggregated.keys(), nds, strict=True)
            }
        )

    def _noisy_fraction(self, count: int, total: int) -> float:
        return float(np.random.normal(count, self.clipped_count_stddev)) / float(total)

    def _geometric_update(self, clipped_fraction: float) -> None:
        self.clipping_norm *= math.exp(
            -self.clip_norm_lr * (clipped_fraction - self.target_clipped_quantile)
        )

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated evaluation."""
        return self.strategy.configure_evaluate(server_round, arrays, config, grid)

    def aggregate_evaluate(
        self, server_round: int, replies: Iterable[Message]
    ) -> MetricRecord | None:
        """Aggregate MetricRecords in the received Messages."""
        return self.strategy.aggregate_evaluate(server_round, replies)

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        self.strategy.summary()


class DifferentialPrivacyServerSideAdaptiveClipping(DifferentialPrivacyAdaptiveBase):
    """Message-based central DP with server-side adaptive clipping."""

    # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
    def __init__(
        self,
        strategy: Strategy,
        noise_multiplier: float,
        num_sampled_clients: int,
        initial_clipping_norm: float = 0.1,
        target_clipped_quantile: float = 0.5,
        clip_norm_lr: float = 0.2,
        clipped_count_stddev: float | None = None,
    ) -> None:
        super().__init__(
            strategy,
            noise_multiplier,
            num_sampled_clients,
            initial_clipping_norm,
            target_clipped_quantile,
            clip_norm_lr,
            clipped_count_stddev,
        )
        self.current_arrays: ArrayRecord = ArrayRecord()

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        return "Differential Privacy Strategy Wrapper (Server-Side Adaptive Clipping)"

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> DP settings:")
        log(INFO, "\t│\t├── Noise multiplier: %s", self.noise_multiplier)
        log(INFO, "\t│\t├── Clipping norm: %s", self.clipping_norm)
        log(INFO, "\t│\t├── Target clipped quantile: %s", self.target_clipped_quantile)
        log(INFO, "\t│\t└── Clip norm learning rate: %s", self.clip_norm_lr)
        super().summary()

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of training."""
        self.current_arrays = arrays
        return self.strategy.configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        if not validate_replies(replies, self.num_sampled_clients):
            return None, None

        current_nd = self.current_arrays.to_numpy_ndarrays()
        clipped_indicator_count = 0
        replies_list = list(replies)

        for reply in replies_list:
            for arr_name, record in reply.content.array_records.items():
                reply_nd = record.to_numpy_ndarrays()
                model_update = [
                    np.subtract(x, y)
                    for (x, y) in zip(reply_nd, current_nd, strict=True)
                ]
                norm_bit = adaptive_clip_inputs_inplace(
                    model_update, self.clipping_norm
                )
                clipped_indicator_count += int(norm_bit)
                # reconstruct array using clipped contribution from current round
                restored = [
                    c + u for c, u in zip(current_nd, model_update, strict=True)
                ]
                reply.content[arr_name] = ArrayRecord(
                    {
                        k: Array(np.asarray(v))
                        for k, v in zip(record.keys(), restored, strict=True)
                    }
                )
            log(
                INFO,
                "aggregate_train: arrays in `ArrayRecord` are clipped by value: %.4f.",
                self.clipping_norm,
            )

        clipped_fraction = self._noisy_fraction(
            clipped_indicator_count, len(replies_list)
        )
        self._geometric_update(clipped_fraction)

        aggregated_arrays, aggregated_metrics = self.strategy.aggregate_train(
            server_round, replies_list
        )

        if aggregated_arrays:
            aggregated_arrays = self._add_noise_to_aggregated_arrays(aggregated_arrays)

        return aggregated_arrays, aggregated_metrics


class DifferentialPrivacyClientSideAdaptiveClipping(DifferentialPrivacyAdaptiveBase):
    """Strategy wrapper for central DP with client-side adaptive clipping.

    Use `adaptiveclipping_mod` modifier at the client side.

    In comparison to `DifferentialPrivacyServerSideAdaptiveClipping`,
    which performs clipping on the server-side,
    `DifferentialPrivacyClientSideAdaptiveClipping`
    expects clipping to happen on the client-side, usually by using the built-in
    `adaptiveclipping_mod`.

    Parameters
    ----------
    strategy : Strategy
        The strategy to which DP functionalities will be added by this wrapper.
    noise_multiplier : float
        The noise multiplier for the Gaussian mechanism for model updates.
    num_sampled_clients : int
        The number of clients that are sampled on each round.
    initial_clipping_norm : float
        The initial value of clipping norm. Defaults to 0.1.
        Andrew et al. recommends to set to 0.1.
    target_clipped_quantile : float
        The desired quantile of updates which should be clipped. Defaults to 0.5.
    clip_norm_lr : float
        The learning rate for the clipping norm adaptation. Defaults to 0.2.
        Andrew et al. recommends to set to 0.2.
    clipped_count_stddev : float
        The stddev of the noise added to the count of
        updates currently below the estimate.
        Andrew et al. recommends to set to `expected_num_records/20`

    Examples
    --------
    Create a strategy::

        strategy = fl.serverapp.FedAvg(...)

    Wrap the strategy with the `DifferentialPrivacyClientSideAdaptiveClipping` wrapper::

        dp_strategy = DifferentialPrivacyClientSideAdaptiveClipping(
            strategy, cfg.noise_multiplier, cfg.num_sampled_clients, ...
        )

    On the client, add the `adaptiveclipping_mod` to the client-side mods::

        app = fl.client.ClientApp(mods=[adaptiveclipping_mod])
    """

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        return "Differential Privacy Strategy Wrapper (Client-Side Adaptive Clipping)"

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> DP settings:")
        log(INFO, "\t│\t├── Noise multiplier: %s", self.noise_multiplier)
        log(INFO, "\t│\t├── Clipping norm: %s", self.clipping_norm)
        log(INFO, "\t│\t├── Target clipped quantile: %s", self.target_clipped_quantile)
        log(INFO, "\t│\t└── Clip norm learning rate: %s", self.clip_norm_lr)
        super().summary()

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of training."""
        config[KEY_CLIPPING_NORM] = self.clipping_norm
        return self.strategy.configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        if not validate_replies(replies, self.num_sampled_clients):
            return None, None

        replies_list = list(replies)

        # validate that KEY_NORM_BIT is present in all replies
        for msg in replies_list:
            for _, mrec in msg.content.metric_records.items():
                if KEY_NORM_BIT not in mrec:
                    raise AggregationError(
                        f"KEY_NORM_BIT ('{KEY_NORM_BIT}') not found"
                        f" in MetricRecord or metrics for reply."
                    )

        aggregated_arrays, aggregated_metrics = self.strategy.aggregate_train(
            server_round, replies_list
        )

        self._update_clip_norm_from_replies(replies_list)

        if aggregated_arrays:
            aggregated_arrays = self._add_noise_to_aggregated_arrays(aggregated_arrays)

        return aggregated_arrays, aggregated_metrics

    def _update_clip_norm_from_replies(self, replies: list[Message]) -> None:
        total = len(replies)
        clipped_count = 0

        for msg in replies:
            # KEY_NORM_BIT is guaranteed to be present
            for _, mrec in msg.content.metric_records.items():
                if KEY_NORM_BIT in mrec:
                    clipped_count += int(bool(mrec[KEY_NORM_BIT]))
                    break
            else:
                # Check fallback location
                if hasattr(msg.content, "metrics") and isinstance(
                    msg.content.metrics, dict
                ):
                    clipped_count += int(bool(msg.content.metrics[KEY_NORM_BIT]))

        clipped_fraction = self._noisy_fraction(clipped_count, total)
        self._geometric_update(clipped_fraction)
