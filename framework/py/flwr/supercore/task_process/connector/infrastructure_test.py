# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
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
"""Tests for shared connector infrastructure."""


from unittest.mock import Mock, patch

import pytest
import requests

from .http import ConnectorApiError, request_json_object
from .json_utils import optional_string
from .registry import CONNECTORS


class ExampleApiError(ConnectorApiError):
    """Test connector error."""

    provider = "Example"


def test_account_connector_input_schemas_are_strict() -> None:
    """Connector schemas should reject unknown and empty string arguments."""
    for connector in CONNECTORS:
        for action in connector.provider.actions:
            schema = action.input_schema
            assert schema["additionalProperties"] is False
            _assert_non_empty_string_schemas(schema)


@pytest.mark.parametrize("value", [None, "", "   "])
def test_optional_string_normalizes_blank_values(value: object) -> None:
    """Blank optional strings should behave like omitted arguments."""
    assert optional_string(value, "Example", "cursor") is None


def test_optional_string_rejects_non_string_values() -> None:
    """Invalid optional string types should not be silently omitted."""
    with pytest.raises(ValueError, match="must be a non-empty string"):
        optional_string(1, "Example", "cursor")


def _assert_non_empty_string_schemas(value: object) -> None:
    """Assert that non-enum strings at any schema depth reject emptiness."""
    if isinstance(value, dict):
        if value.get("type") == "string" and "enum" not in value:
            assert value["minLength"] == 1
        for nested in value.values():
            _assert_non_empty_string_schemas(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_non_empty_string_schemas(nested)


def test_json_request_failure_is_secret_safe() -> None:
    """Transport failures should not expose provider secrets."""
    request = Mock(side_effect=requests.RequestException("secret"))

    with (
        patch("flwr.supercore.task_process.connector.http.requests.request", request),
        pytest.raises(ExampleApiError) as exc_info,
    ):
        request_json_object(
            "GET", "https://api.example.com/items", error=ExampleApiError
        )

    assert exc_info.value.code == "request_failed"
    assert "secret" not in str(exc_info.value)
