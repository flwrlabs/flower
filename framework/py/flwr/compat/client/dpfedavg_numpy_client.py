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
"""Wrapper for configuring a Flower client for DP."""


import copy

import numpy as np

from flwr.common.logger import warn_deprecated_feature
from flwr.compat.client.numpy_client import NumPyClient
from flwr.compat.common.dp import add_gaussian_noise, clip_by_l2
from flwr.compat.common.typing import Config, NDArrays, Scalar


class DPFedAvgNumPyClient(NumPyClient):
    """Wrapper for configuring a Flower client for DP.

    Warning
    -------
    This class is deprecated and will be removed in a future release.
    """

    def __init__(self, client: NumPyClient) -> None:
        warn_deprecated_feature("`DPFedAvgNumPyClient` wrapper")
        super().__init__()
        self.client = client

    def get_properties(self, config: Config) -> dict[str, Scalar]:
        """Get client properties using the given Numpy client.

        Parameters
        ----------
        config : Config
            Configuration parameters requested by the server.
            This can be used to tell the client which properties
            are needed along with some Scalar attributes.

        Returns
        -------
        properties : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of type
            bool, bytes, float, int, or str. It can be used to communicate
            arbitrary property values back to the server.
        """
        return self.client.get_properties(config)

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """Return the current local model parameters.

        Parameters
        ----------
        config : Config
            Configuration parameters requested by the server.
            This can be used to tell the client which parameters
            are needed along with some Scalar attributes.

        Returns
        -------
        parameters : NDArrays
            The local model parameters as a list of NumPy ndarrays.
        """
        return self.client.get_parameters(config)

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Train the provided parameters using the locally held dataset.

        This method first updates the local model using the original parameters
        provided. It then calculates the update by subtracting the original
        parameters from the updated model. The update is then clipped by an L2
        norm and Gaussian noise is added if specified by the configuration.

        The update is then applied to the original parameters to obtain the
        updated parameters which are returned along with the number of examples
        used and metrics computed during the fitting process.

        Parameters
        ----------
        parameters : NDArrays
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the
            server to influence training on the client. It can be used to
            communicate arbitrary values from the server to the client, for
            example, to set the number of (local) training epochs.

        Returns
        -------
        parameters : NDArrays
            The locally updated model parameters.
        num_examples : int
            The number of examples used for training.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of type
            bool, bytes, float, int, or str. It can be used to communicate
            arbitrary values back to the server.

        Raises
        ------
        Exception
            If any required configuration parameters are not provided or are of
            the wrong type.
        """
        original_params = copy.deepcopy(parameters)
        # Getting the updated model from the wrapped client
        updated_params, num_examples, metrics = self.client.fit(parameters, config)

        # Update = updated model - original model
        update = [
            np.subtract(x, y)
            for (x, y) in zip(updated_params, original_params, strict=True)
        ]

        if "dpfedavg_clip_norm" not in config:
            raise KeyError("Clipping threshold not supplied by the server.")
        if not isinstance(config["dpfedavg_clip_norm"], float):
            raise TypeError("Clipping threshold should be a floating point value.")

        # Clipping
        update, clipped = clip_by_l2(update, config["dpfedavg_clip_norm"])

        if "dpfedavg_noise_stddev" in config:
            if not isinstance(config["dpfedavg_noise_stddev"], float):
                raise TypeError(
                    "Scale of noise to be added should be a floating point value."
                )
            # Noising
            update = add_gaussian_noise(update, config["dpfedavg_noise_stddev"])

        for i, _ in enumerate(original_params):
            updated_params[i] = original_params[i] + update[i]

        # Calculating value of norm indicator bit, required for adaptive clipping
        if "dpfedavg_adaptive_clip_enabled" in config:
            if not isinstance(config["dpfedavg_adaptive_clip_enabled"], bool):
                raise TypeError(
                    "dpfedavg_adaptive_clip_enabled should be a boolean-valued flag."
                )
            metrics["dpfedavg_norm_bit"] = not clipped

        return updated_params, num_examples, metrics

    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Evaluate the provided parameters using the locally held dataset.

        Parameters
        ----------
        parameters : NDArrays
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the server to influence
            evaluation on the client. It can be used to communicate
            arbitrary values from the server to the client, for example,
            to influence the number of examples used for evaluation.

        Returns
        -------
        loss : float
            The evaluation loss of the model on the local dataset.
        num_examples : int
            The number of examples used for evaluation.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of
            type bool, bytes, float, int, or str. It can be used to
            communicate arbitrary values back to the server.

        Warning
        -------
        The previous return type format (int, float, float) and the
        extended format (int, float, float, Dict[str, Scalar]) have been
        deprecated and removed since Flower 0.19.
        """
        return self.client.evaluate(parameters, config)
