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
"""Flower client app."""


from abc import ABC
from collections.abc import Callable

from flwr.compat.client.client import Client
from flwr.compat.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.compat.common.typing import (
    Code,
    Config,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    NDArrays,
    Scalar,
    Status,
)

EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_FIT = """
NumPyClient.fit did not return a tuple with 3 elements.
The returned values should have the following type signature:

    Tuple[NDArrays, int, Dict[str, Scalar]]

Example
-------

    model.get_weights(), 10, {"accuracy": 0.95}

"""

EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_EVALUATE = """
NumPyClient.evaluate did not return a tuple with 3 elements.
The returned values should have the following type signature:

    Tuple[float, int, Dict[str, Scalar]]

Example
-------

    0.5, 10, {"accuracy": 0.95}

"""


class NumPyClient(ABC):
    """Abstract base class for Flower clients using NumPy."""

    def get_properties(self, config: Config) -> dict[str, Scalar]:
        """Return a client's set of properties.

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
        _ = (self, config)
        return {}

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
        _ = (self, config)
        return []

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Train the provided parameters using the locally held dataset.

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
        """
        _ = (self, parameters, config)
        return [], 0, {}

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
        _ = (self, parameters, config)
        return 0.0, 0, {}

    def to_client(self) -> Client:
        """Convert to object to Client type and return it."""
        return _wrap_numpy_client(client=self)


def has_get_properties(client: NumPyClient) -> bool:
    """Check if NumPyClient implements get_properties."""
    return type(client).get_properties != NumPyClient.get_properties


def has_get_parameters(client: NumPyClient) -> bool:
    """Check if NumPyClient implements get_parameters."""
    return type(client).get_parameters != NumPyClient.get_parameters


def has_fit(client: NumPyClient) -> bool:
    """Check if NumPyClient implements fit."""
    return type(client).fit != NumPyClient.fit


def has_evaluate(client: NumPyClient) -> bool:
    """Check if NumPyClient implements evaluate."""
    return type(client).evaluate != NumPyClient.evaluate


def _constructor(self: Client, numpy_client: NumPyClient) -> None:
    self.numpy_client = numpy_client  # type: ignore


def _get_properties(self: Client, ins: GetPropertiesIns) -> GetPropertiesRes:
    """Return the current client properties."""
    properties = self.numpy_client.get_properties(config=ins.config)  # type: ignore
    return GetPropertiesRes(
        status=Status(code=Code.OK, message="Success"),
        properties=properties,
    )


def _get_parameters(self: Client, ins: GetParametersIns) -> GetParametersRes:
    """Return the current local model parameters."""
    parameters = self.numpy_client.get_parameters(config=ins.config)  # type: ignore
    parameters_proto = ndarrays_to_parameters(parameters)
    return GetParametersRes(
        status=Status(code=Code.OK, message="Success"), parameters=parameters_proto
    )


def _fit(self: Client, ins: FitIns) -> FitRes:
    """Refine the provided parameters using the locally held dataset."""
    # Deconstruct FitIns
    parameters: NDArrays = parameters_to_ndarrays(ins.parameters)

    # Train
    results = self.numpy_client.fit(parameters, ins.config)  # type: ignore
    if not (
        len(results) == 3
        and isinstance(results[0], list)
        and isinstance(results[1], int)
        and isinstance(results[2], dict)
    ):
        raise TypeError(EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_FIT)

    # Return FitRes
    parameters_prime, num_examples, metrics = results
    parameters_prime_proto = ndarrays_to_parameters(parameters_prime)
    return FitRes(
        status=Status(code=Code.OK, message="Success"),
        parameters=parameters_prime_proto,
        num_examples=num_examples,
        metrics=metrics,
    )


def _evaluate(self: Client, ins: EvaluateIns) -> EvaluateRes:
    """Evaluate the provided parameters using the locally held dataset."""
    parameters: NDArrays = parameters_to_ndarrays(ins.parameters)

    results = self.numpy_client.evaluate(parameters, ins.config)  # type: ignore
    if not (
        len(results) == 3
        and isinstance(results[0], float)
        and isinstance(results[1], int)
        and isinstance(results[2], dict)
    ):
        raise TypeError(EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_EVALUATE)

    # Return EvaluateRes
    loss, num_examples, metrics = results
    return EvaluateRes(
        status=Status(code=Code.OK, message="Success"),
        loss=loss,
        num_examples=num_examples,
        metrics=metrics,
    )


def _wrap_numpy_client(client: NumPyClient) -> Client:
    member_dict: dict[str, Callable] = {  # type: ignore
        "__init__": _constructor,
    }

    # Add wrapper type methods (if overridden)

    if has_get_properties(client=client):
        member_dict["get_properties"] = _get_properties

    if has_get_parameters(client=client):
        member_dict["get_parameters"] = _get_parameters

    if has_fit(client=client):
        member_dict["fit"] = _fit

    if has_evaluate(client=client):
        member_dict["evaluate"] = _evaluate

    # Create wrapper class
    wrapper_class = type("NumPyClientWrapper", (Client,), member_dict)

    # Create and return an instance of the newly created class
    return wrapper_class(numpy_client=client)  # type: ignore
