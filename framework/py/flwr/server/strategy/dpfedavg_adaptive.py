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
"""DP-FedAvg [Andrew et al., 2019] with adaptive clipping.

Paper: arxiv.org/pdf/1905.03871.pdf
"""


import math

import numpy as np

from flwr.common import FitIns, FitRes, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.dpfedavg_fixed import DPFedAvgFixed
from flwr.server.strategy.strategy import Strategy
from flwr.supercore.logger import warn_deprecated_feature


class DPFedAvgAdaptive(DPFedAvgFixed):
    """Wrapper for configuring a Strategy for DP with Adaptive Clipping.

    Warning
    -------
    This class is deprecated and will be removed in a future release.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-positional-arguments
    def __init__(
        self,
        strategy: Strategy,
        num_sampled_clients: int,
        init_clip_norm: float = 0.1,
        noise_multiplier: float = 1.0,
        server_side_noising: bool = True,
        clip_norm_lr: float = 0.2,
        clip_norm_target_quantile: float = 0.5,
        clip_count_stddev: float | None = None,
    ) -> None:
        warn_deprecated_feature("`DPFedAvgAdaptive` wrapper")
        super().__init__(
            strategy=strategy,
            num_sampled_clients=num_sampled_clients,
            clip_norm=init_clip_norm,
            noise_multiplier=noise_multiplier,
            server_side_noising=server_side_noising,
        )
        self.clip_norm_lr = clip_norm_lr
        self.clip_norm_target_quantile = clip_norm_target_quantile

        if clip_count_stddev is None:
            clip_count_stddev = 0.0
            if noise_multiplier > 0:
                clip_count_stddev = self.num_sampled_clients / 20.0
        self.clip_count_stddev: float = clip_count_stddev

        if noise_multiplier:
            self.noise_multiplier = (
                self.noise_multiplier ** (-2) - (2 * self.clip_count_stddev) ** (-2)
            ) ** (-0.5)

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = "Strategy with DP with Adaptive Clipping enabled."
        return rep

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        additional_config = {"dpfedavg_adaptive_clip_enabled": True}

        client_instructions = super().configure_fit(
            server_round, parameters, client_manager
        )

        for _, fit_ins in client_instructions:
            fit_ins.config.update(additional_config)

        return client_instructions

    def _update_clip_norm(self, results: list[tuple[ClientProxy, FitRes]]) -> None:
        # Calculating number of clients which set the norm indicator bit
        norm_bit_set_count = 0
        for client_proxy, fit_res in results:
            if "dpfedavg_norm_bit" not in fit_res.metrics:
                raise KeyError(
                    f"Indicator bit not returned by client with id {client_proxy.cid}."
                )
            if fit_res.metrics["dpfedavg_norm_bit"]:
                norm_bit_set_count += 1
        # Noising the count
        noised_norm_bit_set_count = float(
            np.random.normal(norm_bit_set_count, self.clip_count_stddev)
        )

        noised_norm_bit_set_fraction = noised_norm_bit_set_count / len(results)
        # Geometric update
        self.clip_norm *= math.exp(
            -self.clip_norm_lr
            * (noised_norm_bit_set_fraction - self.clip_norm_target_quantile)
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate training results as in DPFedAvgFixed and update clip norms."""
        if failures:
            return None, {}
        new_global_model = super().aggregate_fit(server_round, results, failures)
        self._update_clip_norm(results)
        return new_global_model
