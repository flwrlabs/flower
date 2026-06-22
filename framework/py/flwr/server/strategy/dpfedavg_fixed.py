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
"""DP-FedAvg [McMahan et al., 2018] strategy.

Paper: arxiv.org/pdf/1710.06963.pdf
"""


from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import warn_deprecated_feature
from flwr.compat.common.dp import add_gaussian_noise
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.strategy import Strategy


class DPFedAvgFixed(Strategy):
    """Wrapper for configuring a Strategy for DP with Fixed Clipping.

    Warning
    -------
    This class is deprecated and will be removed in a future release.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-positional-arguments
    def __init__(
        self,
        strategy: Strategy,
        num_sampled_clients: int,
        clip_norm: float,
        noise_multiplier: float = 1,
        server_side_noising: bool = True,
    ) -> None:
        warn_deprecated_feature("`DPFedAvgFixed` wrapper")
        super().__init__()
        self.strategy = strategy
        # Doing fixed-size subsampling as in https://arxiv.org/abs/1905.03871.
        self.num_sampled_clients = num_sampled_clients

        if clip_norm <= 0:
            raise ValueError("The clipping threshold should be a positive value.")
        self.clip_norm = clip_norm

        if noise_multiplier < 0:
            raise ValueError("The noise multiplier should be a non-negative value.")
        self.noise_multiplier = noise_multiplier

        self.server_side_noising = server_side_noising

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = "Strategy with DP with Fixed Clipping enabled."
        return rep

    def _calc_client_noise_stddev(self) -> float:
        return float(
            self.noise_multiplier * self.clip_norm / (self.num_sampled_clients ** (0.5))
        )

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        """Initialize global model parameters using given strategy."""
        return self.strategy.initialize_parameters(client_manager)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training incorporating Differential Privacy (DP).

        Configuration of the next training round includes information related to DP,
        such as clip norm and noise stddev.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        parameters : Parameters
            The current (global) model parameters.
        client_manager : ClientManager
            The client manager which holds all currently connected clients.

        Returns
        -------
        fit_configuration : List[Tuple[ClientProxy, FitIns]]
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `FitIns` for this particular `ClientProxy`. If a particular `ClientProxy`
            is not included in this list, it means that this `ClientProxy`
            will not participate in the next round of federated learning.
        """
        additional_config = {"dpfedavg_clip_norm": self.clip_norm}
        if not self.server_side_noising:
            additional_config["dpfedavg_noise_stddev"] = (
                self._calc_client_noise_stddev()
            )

        client_instructions = self.strategy.configure_fit(
            server_round, parameters, client_manager
        )

        for _, fit_ins in client_instructions:
            fit_ins.config.update(additional_config)

        return client_instructions

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation using the specified strategy.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        parameters : Parameters
            The current (global) model parameters.
        client_manager : ClientManager
            The client manager which holds all currently connected clients.

        Returns
        -------
        evaluate_configuration : List[Tuple[ClientProxy, EvaluateIns]]
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `EvaluateIns` for this particular `ClientProxy`. If a particular
            `ClientProxy` is not included in this list, it means that this
            `ClientProxy` will not participate in the next round of federated
            evaluation.
        """
        return self.strategy.configure_evaluate(
            server_round, parameters, client_manager
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate training results using unweighted aggregation."""
        if failures:
            return None, {}
        # Forcing unweighted aggregation, as in https://arxiv.org/abs/1905.03871.
        for _, fit_res in results:
            fit_res.num_examples = 1
            fit_res.parameters = ndarrays_to_parameters(
                add_gaussian_noise(
                    parameters_to_ndarrays(fit_res.parameters),
                    self._calc_client_noise_stddev(),
                )
            )

        return self.strategy.aggregate_fit(server_round, results, failures)

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
