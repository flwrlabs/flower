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


from collections.abc import Mapping
from unittest.mock import Mock, patch

import pytest
import requests

from flwr.supercore.typing import JSONObject

from .http import ConnectorApiError, request_json_object
from .oauth import BaseOAuthProvider, load_oauth_provider


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


class ExampleOAuthProvider(BaseOAuthProvider):
    """Minimal provider for the shared OAuth flow."""

    display_name = "Example"
    authorize_url = "https://example.com/authorize"
    error_type = RuntimeError

    def __init__(self, response: Mock | None = None, **kwargs: str) -> None:
        super().__init__(**kwargs)
        self.response = response or Mock()

    def authorization_parameters(
        self,
        *,
        redirect_uri: str,
        state: str,
        pkce_challenge: str | None,
    ) -> Mapping[str, str]:
        """Return test authorization parameters."""
        del pkce_challenge
        return {"redirect_uri": redirect_uri, "state": state}

    def request_token(
        self,
        *,
        code: str,
        redirect_uri: str,
        pkce_verifier: str | None,
    ) -> requests.Response:
        """Return the configured token response."""
        del code, redirect_uri, pkce_verifier
        return self.response

    def parse_token_response(
        self, payload: JSONObject
    ) -> tuple[JSONObject, JSONObject]:
        """Return credentials from the token response."""
        return {"access_token": str(payload["access_token"])}, {}


def test_oauth_flow_and_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """OAuth should parse tokens and reject invalid environment configuration."""
    response = Mock(status_code=200)
    response.json.return_value = {"access_token": "token"}
    provider = ExampleOAuthProvider(
        response,
        client_id="client",
        client_secret="secret",
        redirect_uri="https://example.com/callback",
    )
    assert provider.exchange_code(
        code="code",
        redirect_uri="https://example.com/callback",
        pkce_verifier=None,
    ) == ({"access_token": "token"}, {})

    monkeypatch.setenv("EXAMPLE_CLIENT_ID", "client")
    monkeypatch.delenv("EXAMPLE_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("EXAMPLE_REDIRECT_URI", raising=False)
    with pytest.raises(ValueError, match="configuration is incomplete"):
        load_oauth_provider(
            ExampleOAuthProvider,
            client_id_env="EXAMPLE_CLIENT_ID",
            client_secret_env="EXAMPLE_CLIENT_SECRET",
            redirect_uri_env="EXAMPLE_REDIRECT_URI",
        )
