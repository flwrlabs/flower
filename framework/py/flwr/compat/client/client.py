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
"""Flower client (abstract base class)."""


# Needed to `Client` class can return a type of `Client` (not needed in py3.11+)
from __future__ import annotations

from abc import ABC

from flwr.compat.common.typing import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    Status,
)


class Client(ABC):
    """Abstract base class for Flower clients."""

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        """Return set of client's properties.

        Parameters
        ----------
        ins : GetPropertiesIns
            The get properties instructions received from the server containing
            a dictionary of configuration values.

        Returns
        -------
        GetPropertiesRes
            The current client properties.
        """
        _ = (self, ins)
        return GetPropertiesRes(
            status=Status(
                code=Code.GET_PROPERTIES_NOT_IMPLEMENTED,
                message="Client does not implement `get_properties`",
            ),
            properties={},
        )

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Return the current local model parameters.

        Parameters
        ----------
        ins : GetParametersIns
            The get parameters instructions received from the server containing
            a dictionary of configuration values.

        Returns
        -------
        GetParametersRes
            The current local model parameters.
        """
        _ = (self, ins)
        return GetParametersRes(
            status=Status(
                code=Code.GET_PARAMETERS_NOT_IMPLEMENTED,
                message="Client does not implement `get_parameters`",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def fit(self, ins: FitIns) -> FitRes:
        """Refine the provided parameters using the locally held dataset.

        Parameters
        ----------
        ins : FitIns
            The training instructions containing (global) model parameters
            received from the server and a dictionary of configuration values
            used to customize the local training process.

        Returns
        -------
        FitRes
            The training result containing updated parameters and other details
            such as the number of local training examples used for training.
        """
        _ = (self, ins)
        return FitRes(
            status=Status(
                code=Code.FIT_NOT_IMPLEMENTED,
                message="Client does not implement `fit`",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
            num_examples=0,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate the provided parameters using the locally held dataset.

        Parameters
        ----------
        ins : EvaluateIns
            The evaluation instructions containing (global) model parameters
            received from the server and a dictionary of configuration values
            used to customize the local evaluation process.

        Returns
        -------
        EvaluateRes
            The evaluation result containing the loss on the local dataset and
            other details such as the number of local data examples used for
            evaluation.
        """
        _ = (self, ins)
        return EvaluateRes(
            status=Status(
                code=Code.EVALUATE_NOT_IMPLEMENTED,
                message="Client does not implement `evaluate`",
            ),
            loss=0.0,
            num_examples=0,
            metrics={},
        )

    def to_client(self) -> Client:
        """Return client (itself)."""
        return self


def has_get_properties(client: Client) -> bool:
    """Check if Client implements get_properties."""
    return type(client).get_properties != Client.get_properties


def has_get_parameters(client: Client) -> bool:
    """Check if Client implements get_parameters."""
    return type(client).get_parameters != Client.get_parameters


def has_fit(client: Client) -> bool:
    """Check if Client implements fit."""
    return type(client).fit != Client.fit


def has_evaluate(client: Client) -> bool:
    """Check if Client implements evaluate."""
    return type(client).evaluate != Client.evaluate


def maybe_call_get_properties(
    client: Client, get_properties_ins: GetPropertiesIns
) -> GetPropertiesRes:
    """Call `get_properties` if the client overrides it."""
    # Check if client overrides `get_properties`
    if not has_get_properties(client=client):
        # If client does not override `get_properties`, don't call it
        status = Status(
            code=Code.GET_PROPERTIES_NOT_IMPLEMENTED,
            message="Client does not implement `get_properties`",
        )
        return GetPropertiesRes(
            status=status,
            properties={},
        )

    # If the client implements `get_properties`, call it
    return client.get_properties(get_properties_ins)


def maybe_call_get_parameters(
    client: Client, get_parameters_ins: GetParametersIns
) -> GetParametersRes:
    """Call `get_parameters` if the client overrides it."""
    # Check if client overrides `get_parameters`
    if not has_get_parameters(client=client):
        # If client does not override `get_parameters`, don't call it
        status = Status(
            code=Code.GET_PARAMETERS_NOT_IMPLEMENTED,
            message="Client does not implement `get_parameters`",
        )
        return GetParametersRes(
            status=status,
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    # If the client implements `get_parameters`, call it
    return client.get_parameters(get_parameters_ins)


def maybe_call_fit(client: Client, fit_ins: FitIns) -> FitRes:
    """Call `fit` if the client overrides it."""
    # Check if client overrides `fit`
    if not has_fit(client=client):
        # If client does not override `fit`, don't call it
        status = Status(
            code=Code.FIT_NOT_IMPLEMENTED,
            message="Client does not implement `fit`",
        )
        return FitRes(
            status=status,
            parameters=Parameters(tensor_type="", tensors=[]),
            num_examples=0,
            metrics={},
        )

    # If the client implements `fit`, call it
    return client.fit(fit_ins)


def maybe_call_evaluate(client: Client, evaluate_ins: EvaluateIns) -> EvaluateRes:
    """Call `evaluate` if the client overrides it."""
    # Check if client overrides `evaluate`
    if not has_evaluate(client=client):
        # If client does not override `evaluate`, don't call it
        status = Status(
            code=Code.EVALUATE_NOT_IMPLEMENTED,
            message="Client does not implement `evaluate`",
        )
        return EvaluateRes(
            status=status,
            loss=0.0,
            num_examples=0,
            metrics={},
        )

    # If the client implements `evaluate`, call it
    return client.evaluate(evaluate_ins)
