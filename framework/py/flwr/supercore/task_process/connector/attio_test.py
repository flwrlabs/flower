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
"""Tests for read-only Attio connector tools."""

from unittest.mock import Mock, patch

import pytest

from . import attio, registry
from .attio import (
    ATTIO_API_BASE_URL,
    ATTIO_CONNECTOR_REF,
    AttioApiError,
    make_attio_tools,
    search_records,
)
from flwr.supercore.typing import JSONObject

_CREDENTIALS: JSONObject = {"access_token": "attio-secret"}


def _response(payload: object, *, status_code: int = 200) -> Mock:
    response = Mock(status_code=status_code)
    response.json.return_value = payload
    return response


def test_attio_tools_are_registered_as_read_only_credentials() -> None:
    """Attio tools should resolve through the shared connector registry."""
    tools = registry.get_connector_tools(ATTIO_CONNECTOR_REF)

    assert [tool["name"] for tool in tools] == list(attio.ATTIO_TOOL_HANDLERS)
    assert tools == make_attio_tools()
    for tool in tools:
        name = str(tool["name"])
        parameters = tool["parameters"]
        assert isinstance(parameters, dict)
        assert parameters["additionalProperties"] is False
        assert "create" not in name and "update" not in name
        assert registry.requires_connector_credentials(name)
        assert registry.get_connector_ref(name) == ATTIO_CONNECTOR_REF


def test_search_records_calls_attio() -> None:
    """Record search should pass the documented request to Attio."""
    response = _response({"data": []})
    with patch(
        "flwr.supercore.task_process.connector.attio.requests.request",
        return_value=response,
    ) as request:
        result = search_records(
            "Flower",
            ["companies"],
            credentials=_CREDENTIALS,
            config={},
            usage_recorder=Mock(),
        )

    request.assert_called_once_with(
        "POST",
        f"{ATTIO_API_BASE_URL}/objects/records/search",
        headers={
            "Authorization": "Bearer attio-secret",
            "Content-Type": "application/json",
        },
        params=None,
        json={"query": "Flower", "objects": ["companies"], "limit": 25},
        timeout=30.0,
    )
    assert result == {"data": []}


def test_api_errors_are_stable_and_secret_safe() -> None:
    """API errors should not expose tokens or response text."""
    with (
        patch(
            "flwr.supercore.task_process.connector.attio.requests.request",
            return_value=_response(
                {"code": "permission_denied", "message": "attio-secret"},
                status_code=401,
            ),
        ),
        pytest.raises(AttioApiError) as error,
    ):
        search_records(
            "Flower",
            ["companies"],
            credentials=_CREDENTIALS,
            config={},
            usage_recorder=Mock(),
        )

    assert error.value.code == "http_error"
    assert "attio-secret" not in str(error.value)
