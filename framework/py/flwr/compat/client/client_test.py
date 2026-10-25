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
"""Flower Client tests."""


from unittest.mock import MagicMock

from flwr.common import (
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

from .client import (
    Client,
    has_evaluate,
    has_fit,
    has_get_parameters,
    has_get_properties,
    maybe_call_evaluate,
    maybe_call_fit,
    maybe_call_get_parameters,
    maybe_call_get_properties,
)


class OverridingClient(Client):
    """Client overriding `get_properties`."""

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        """Get empty properties of the client with 'Success' status."""
        return GetPropertiesRes(
            status=Status(code=Code.OK, message="Success"),
            properties={},
        )

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Get empty parameters of the client with 'Success' status."""
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=Parameters(tensors=[], tensor_type=""),
        )

    def fit(self, ins: FitIns) -> FitRes:
        """Simulate successful training, return no parameters, no metrics."""
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=Parameters(tensors=[], tensor_type=""),
            num_examples=1,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Simulate successful evaluation, return no metrics."""
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=1.0,
            num_examples=1,
            metrics={},
        )


class NotOverridingClient(Client):
    """Client not overriding any Client method."""


def test_has_get_properties_true() -> None:
    """Test has_get_properties."""
    # Prepare
    client = OverridingClient()
    expected = True

    # Execute
    actual = has_get_properties(client=client)

    # Assert
    assert actual == expected


def test_has_get_properties_false() -> None:
    """Test has_get_properties."""
    # Prepare
    client = NotOverridingClient()
    expected = False

    # Execute
    actual = has_get_properties(client=client)

    # Assert
    assert actual == expected


def test_has_get_parameters_true() -> None:
    """Test has_get_parameters."""
    # Prepare
    client = OverridingClient()
    expected = True

    # Execute
    actual = has_get_parameters(client=client)

    # Assert
    assert actual == expected


def test_has_get_parameters_false() -> None:
    """Test has_get_parameters."""
    # Prepare
    client = NotOverridingClient()
    expected = False

    # Execute
    actual = has_get_parameters(client=client)

    # Assert
    assert actual == expected


def test_has_fit_true() -> None:
    """Test has_fit."""
    # Prepare
    client = OverridingClient()
    expected = True

    # Execute
    actual = has_fit(client=client)

    # Assert
    assert actual == expected


def test_has_fit_false() -> None:
    """Test has_fit."""
    # Prepare
    client = NotOverridingClient()
    expected = False

    # Execute
    actual = has_fit(client=client)

    # Assert
    assert actual == expected


def test_has_evaluate_true() -> None:
    """Test has_evaluate."""
    # Prepare
    client = OverridingClient()
    expected = True

    # Execute
    actual = has_evaluate(client=client)

    # Assert
    assert actual == expected


def test_has_evaluate_false() -> None:
    """Test has_evaluate."""
    # Prepare
    client = NotOverridingClient()
    expected = False

    # Execute
    actual = has_evaluate(client=client)

    # Assert
    assert actual == expected


def test_maybe_call_get_properties_true() -> None:
    """Test maybe_call_get_properties."""
    # Prepare
    client = OverridingClient()

    # Execute
    actual = maybe_call_get_properties(client, MagicMock())

    # Assert
    assert actual.status.code == Code.OK


def test_maybe_call_get_properties_false() -> None:
    """Test maybe_call_get_properties."""
    # Prepare
    client = NotOverridingClient()

    # Execute
    actual = maybe_call_get_properties(client, MagicMock())

    # Assert
    assert actual.status.code == Code.GET_PROPERTIES_NOT_IMPLEMENTED


def test_maybe_call_get_parameters_true() -> None:
    """Test maybe_call_get_parameters."""
    # Prepare
    client = OverridingClient()

    # Execute
    actual = maybe_call_get_parameters(client, MagicMock())

    # Assert
    assert actual.status.code == Code.OK


def test_maybe_call_get_parameters_false() -> None:
    """Test maybe_call_get_parameters."""
    # Prepare
    client = NotOverridingClient()

    # Execute
    actual = maybe_call_get_parameters(client, MagicMock())

    # Assert
    assert actual.status.code == Code.GET_PARAMETERS_NOT_IMPLEMENTED


def test_maybe_call_fit_true() -> None:
    """Test maybe_call_fit."""
    # Prepare
    client = OverridingClient()

    # Execute
    actual = maybe_call_fit(client, MagicMock())

    # Assert
    assert actual.status.code == Code.OK


def test_maybe_call_fit_false() -> None:
    """Test maybe_call_fit."""
    # Prepare
    client = NotOverridingClient()

    # Execute
    actual = maybe_call_fit(client, MagicMock())

    # Assert
    assert actual.status.code == Code.FIT_NOT_IMPLEMENTED


def test_maybe_call_evaluate_true() -> None:
    """Test maybe_call_evaluate."""
    # Prepare
    client = OverridingClient()

    # Execute
    actual = maybe_call_evaluate(client, MagicMock())

    # Assert
    assert actual.status.code == Code.OK


def test_maybe_call_evaluate_false() -> None:
    """Test maybe_call_evaluate."""
    # Prepare
    client = NotOverridingClient()

    # Execute
    actual = maybe_call_evaluate(client, MagicMock())

    # Assert
    assert actual.status.code == Code.EVALUATE_NOT_IMPLEMENTED
