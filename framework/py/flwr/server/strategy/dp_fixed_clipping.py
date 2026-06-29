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
"""Central differential privacy with fixed clipping.

Papers: https://arxiv.org/abs/1712.07557, https://arxiv.org/abs/1710.06963
"""


from logging import INFO, WARNING

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
    add_gaussian_noise_to_params,
    compute_clip_model_update,
    compute_stdv,
)


class DifferentialPrivacyServerSideFixedClipping(Strategy):
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

        strategy = fl.server.strategy.FedAvg( ... )

    Wrap the strategy with the DifferentialPrivacyServerSideFixedClipping wrapper::

        dp_strategy = DifferentialPrivacyServerSideFixedClipping(
            strategy, cfg.noise_multiplier, cfg.clipping_norm, cfg.num_sampled_clients
        )
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

        self.current_round_params: NDArrays = []

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = "Differential Privacy Strategy Wrapper (Server-Side Fixed Clipping)"
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
        """Compute the updates, clip, and pass them for aggregation.

        Afterward, add noise to the aggregated parameters.
        """
        if failures:
            return None, {}

        if len(results) != self.num_sampled_clients:
            log(
                WARNING,
                CLIENTS_DISCREPANCY_WARNING,
                len(results),
                self.num_sampled_clients,
            )
        for _, res in results:
            param = parameters_to_ndarrays(res.parameters)
            # Compute and clip update
            compute_clip_model_update(
                param, self.current_round_params, self.clipping_norm
            )
            log(
                INFO,
                "aggregate_fit: parameters are clipped by value: %.4f.",
                self.clipping_norm,
            )
            # Convert back to parameters
            res.parameters = ndarrays_to_parameters(param)

        # Pass the new parameters for aggregation
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


class DifferentialPrivacyClientSideFixedClipping(Strategy):
    """Strategy wrapper for central DP with client-side fixed clipping.

    Use `fixedclipping_mod` modifier at the client side.

    In comparison to `DifferentialPrivacyServerSideFixedClipping`,
    which performs clipping on the server-side, `DifferentialPrivacyClientSideFixedClipping`
    expects clipping to happen on the client-side, usually by using the built-in
    `fixedclipping_mod`.

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

        strategy = fl.server.strategy.FedAvg(...)

    Wrap the strategy with the `DifferentialPrivacyClientSideFixedClipping` wrapper::

        dp_strategy = DifferentialPrivacyClientSideFixedClipping(
            strategy, cfg.noise_multiplier, cfg.clipping_norm, cfg.num_sampled_clients
        )

    On the client, add the `fixedclipping_mod` to the client-side mods::

        app = fl.client.ClientApp(
            client_fn=client_fn, mods=[fixedclipping_mod]
        )
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
            raise ValueError("The clipping threshold should be a positive value.")

        if num_sampled_clients <= 0:
            raise ValueError(
                "The number of sampled clients should be a positive value."
            )

        self.noise_multiplier = noise_multiplier
        self.clipping_norm = clipping_norm
        self.num_sampled_clients = num_sampled_clients

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = "Differential Privacy Strategy Wrapper (Client-Side Fixed Clipping)"
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
        """Add noise to the aggregated parameters."""
        if failures:
            return None, {}

        if len(results) != self.num_sampled_clients:
            log(
                WARNING,
                CLIENTS_DISCREPANCY_WARNING,
                len(results),
                self.num_sampled_clients,
            )

        # Pass the new parameters for aggregation
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
