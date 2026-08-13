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

from .http import ConnectorApiError, ConnectorHttpClient, request_json_object


class ExampleApiError(ConnectorApiError):
    """Test connector error."""

    provider = "Example"


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


def test_connector_http_client_adds_credentials() -> None:
    """Connector HTTP clients should add credentials and provider headers."""
    response = Mock(status_code=200)
    response.json.return_value = {"items": []}
    client = ConnectorHttpClient(
        provider="Example",
        base_url="https://api.example.com/v1",
        credentials={"access_token": "secret"},
        headers={"X-Api-Version": "1"},
    )
    with patch(
        "flwr.supercore.task_process.connector.http.requests.request",
        return_value=response,
    ) as request:
        assert client.request("GET", "/items") == {"items": []}

    assert request.call_args.args == ("GET", "https://api.example.com/v1/items")
    assert request.call_args.kwargs["headers"] == {
        "Authorization": "Bearer secret",
        "X-Api-Version": "1",
    }
