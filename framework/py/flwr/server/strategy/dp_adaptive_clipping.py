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
"""Central differential privacy with adaptive clipping.

Paper (Andrew et al.): https://arxiv.org/abs/1905.03871
"""


import math
from logging import INFO, WARNING

import numpy as np

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.strategy import Strategy
from flwr.supercore.differential_privacy import (
    CLIENTS_DISCREPANCY_WARNING,
    KEY_CLIPPING_NORM,
    KEY_NORM_BIT,
    adaptive_clip_inputs_inplace,
    add_gaussian_noise_to_params,
    compute_adaptive_noise_params,
    compute_stdv,
)


class DifferentialPrivacyServerSideAdaptiveClipping(Strategy):
    """Strategy wrapper for central DP with server-side adaptive clipping.

    Parameters
    ----------
    strategy: Strategy
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
        The standard deviation of the noise added to the count of updates below the estimate.
        Andrew et al. recommends to set to `expected_num_records/20`

    Examples
    --------
    Create a strategy::

        strategy = fl.server.strategy.FedAvg( ... )

    Wrap the strategy with the DifferentialPrivacyServerSideAdaptiveClipping wrapper::

        dp_strategy = DifferentialPrivacyServerSideAdaptiveClipping(
            strategy, cfg.noise_multiplier, cfg.num_sampled_clients, ...
        )
    """

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
            raise ValueError(
                "The target clipped quantile must be between 0 and 1 (inclusive)."
            )

        if clip_norm_lr <= 0:
            raise ValueError("The learning rate must be positive.")

        if clipped_count_stddev is not None:
            if clipped_count_stddev < 0:
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

        self.current_round_params: NDArrays = []

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = "Differential Privacy Strategy Wrapper (Server-Side Adaptive Clipping)"
        return rep

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        """Initialize global model parameters using given strategy."""
        return self.strategy.initialize_parameters(client_manager)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        self.current_round_params = parameters_to_ndarrays(parameters)
        return self.strategy.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        return self.strategy.configure_evaluate(
            server_round, parameters, client_manager
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate training results and update clip norms."""
        if failures:
            return None, {}

        if len(results) != self.num_sampled_clients:
            log(
                WARNING,
                CLIENTS_DISCREPANCY_WARNING,
                len(results),
                self.num_sampled_clients,
            )

        norm_bit_set_count = 0
        for _, res in results:
            param = parameters_to_ndarrays(res.parameters)
            # Compute and clip update
            model_update = [
                np.subtract(x, y)
                for (x, y) in zip(param, self.current_round_params, strict=True)
            ]

            norm_bit = adaptive_clip_inputs_inplace(model_update, self.clipping_norm)
            norm_bit_set_count += norm_bit

            log(
                INFO,
                "aggregate_fit: parameters are clipped by value: %.4f.",
                self.clipping_norm,
            )

            for i, _ in enumerate(self.current_round_params):
                param[i] = self.current_round_params[i] + model_update[i]
            # Convert back to parameters
            res.parameters = ndarrays_to_parameters(param)

        # Noising the count
        noised_norm_bit_set_count = float(
            np.random.normal(norm_bit_set_count, self.clipped_count_stddev)
        )
        noised_norm_bit_set_fraction = noised_norm_bit_set_count / len(results)
        # Geometric update
        self.clipping_norm *= math.exp(
            -self.clip_norm_lr
            * (noised_norm_bit_set_fraction - self.target_clipped_quantile)
        )

        aggregated_params, metrics = self.strategy.aggregate_fit(
            server_round, results, failures
        )

        # Add Gaussian noise to the aggregated parameters
        if aggregated_params:
            aggregated_params = add_gaussian_noise_to_params(
                aggregated_params,
                self.noise_multiplier,
                self.clipping_norm,
                self.num_sampled_clients,
            )
            log(
                INFO,
                "aggregate_fit: central DP noise with %.4f stdev added",
                compute_stdv(
                    self.noise_multiplier, self.clipping_norm, self.num_sampled_clients
                ),
            )

        return aggregated_params, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        """Aggregate evaluation losses using the given strategy."""
        return self.strategy.aggregate_evaluate(server_round, results, failures)

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, Scalar]] | None:
        """Evaluate model parameters using an evaluation function from the strategy."""
        return self.strategy.evaluate(server_round, parameters)


class DifferentialPrivacyClientSideAdaptiveClipping(Strategy):
    """Strategy wrapper for central DP with client-side adaptive clipping.

    Use `adaptiveclipping_mod` modifier at the client side.

    In comparison to `DifferentialPrivacyServerSideAdaptiveClipping`,
    which performs clipping on the server-side, `DifferentialPrivacyClientSideAdaptiveClipping`
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
        The stddev of the noise added to the count of updates currently below the estimate.
        Andrew et al. recommends to set to `expected_num_records/20`

    Examples
    --------
    Create a strategy::

        strategy = fl.server.strategy.FedAvg(...)

    Wrap the strategy with the `DifferentialPrivacyClientSideAdaptiveClipping` wrapper::

        dp_strategy = DifferentialPrivacyClientSideAdaptiveClipping(
            strategy, cfg.noise_multiplier, cfg.num_sampled_clients
        )

    On the client, add the `adaptiveclipping_mod` to the client-side mods::

        app = fl.client.ClientApp(
            client_fn=client_fn, mods=[adaptiveclipping_mod]
        )
    """

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
            raise ValueError(
                "The target clipped quantile must be between 0 and 1 (inclusive)."
            )

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

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = "Differential Privacy Strategy Wrapper (Client-Side Adaptive Clipping)"
        return rep

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        """Initialize global model parameters using given strategy."""
        return self.strategy.initialize_parameters(client_manager)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        additional_config = {KEY_CLIPPING_NORM: self.clipping_norm}
        inner_strategy_config_result = self.strategy.configure_fit(
            server_round, parameters, client_manager
        )
        for _, fit_ins in inner_strategy_config_result:
            fit_ins.config.update(additional_config)

        return inner_strategy_config_result

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        return self.strategy.configure_evaluate(
            server_round, parameters, client_manager
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate training results and update clip norms."""
        if failures:
            return None, {}

        if len(results) != self.num_sampled_clients:
            log(
                WARNING,
                CLIENTS_DISCREPANCY_WARNING,
                len(results),
                self.num_sampled_clients,
            )

        aggregated_params, metrics = self.strategy.aggregate_fit(
            server_round, results, failures
        )
        self._update_clip_norm(results)

        # Add Gaussian noise to the aggregated parameters
        if aggregated_params:
            aggregated_params = add_gaussian_noise_to_params(
                aggregated_params,
                self.noise_multiplier,
                self.clipping_norm,
                self.num_sampled_clients,
            )
            log(
                INFO,
                "aggregate_fit: central DP noise with %.4f stdev added",
                compute_stdv(
                    self.noise_multiplier, self.clipping_norm, self.num_sampled_clients
                ),
            )

        return aggregated_params, metrics

    def _update_clip_norm(self, results: list[tuple[ClientProxy, FitRes]]) -> None:
        # Calculate the number of clients which set the norm indicator bit
        norm_bit_set_count = 0
        for client_proxy, fit_res in results:
            if KEY_NORM_BIT not in fit_res.metrics:
                raise KeyError(
                    f"{KEY_NORM_BIT} not returned by client with id {client_proxy.cid}."
                )
            if fit_res.metrics[KEY_NORM_BIT]:
                norm_bit_set_count += 1
        # Add noise to the count
        noised_norm_bit_set_count = float(
            np.random.normal(norm_bit_set_count, self.clipped_count_stddev)
        )

        noised_norm_bit_set_fraction = noised_norm_bit_set_count / len(results)
        # Geometric update
        self.clipping_norm *= math.exp(
            -self.clip_norm_lr
            * (noised_norm_bit_set_fraction - self.target_clipped_quantile)
        )

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        """Aggregate evaluation losses using the given strategy."""
        return self.strategy.aggregate_evaluate(server_round, results, failures)

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, Scalar]] | None:
        """Evaluate model parameters using an evaluation function from the strategy."""
        return self.strategy.evaluate(server_round, parameters)
