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
# ===============================================================================
"""Tests for the Attio connector."""

from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from flwr.supercore.typing import JSONObject

from .. import registry
from ..definition import ActionAccess
from .actions import ACTIONS
from .executors import AttioApiError
from .oauth import ATTIO_CONNECTOR_REF, AttioOAuthProvider

_HTTP_REQUEST = "flwr.supercore.task_process.connector.http.requests.request"
_TOKEN_REQUEST = "flwr.supercore.task_process.connector.attio.oauth.requests.post"
_CREDENTIALS: JSONObject = {"access_token": "attio-secret"}
_REDIRECT_URI = "https://client.example/oauth/attio"


def _response(payload: object, status_code: int = 200) -> Mock:
    """Return a minimal HTTP response mock."""
    response = Mock(status_code=status_code)
    response.json.return_value = payload
    return response


def _provider() -> AttioOAuthProvider:
    return AttioOAuthProvider(
        client_id="client-id",
        client_secret="client-secret",
        redirect_uri=_REDIRECT_URI,
    )


def test_attio_actions_are_registered_as_read_only() -> None:
    """Attio should expose four account-scoped read actions."""
    assert len(ACTIONS) == 4
    assert all(action.access is ActionAccess.READ for action in ACTIONS)
    tools = registry.get_connector_tools(ATTIO_CONNECTOR_REF)
    assert [tool["name"] for tool in tools] == [
        "attio_search_records",
        "attio_list_meetings",
        "attio_list_call_recordings",
        "attio_get_call_transcript",
    ]


def test_search_records_calls_attio() -> None:
    """Record search should pass the documented request to Attio."""
    response = _response({"data": []})
    with patch(_HTTP_REQUEST, return_value=response) as request:
        result = registry.invoke_connector(
            "attio_search_records",
            {"query": "Flower", "objects": ["companies"]},
            Mock(),
            _CREDENTIALS,
            {},
        )

    assert request.call_args.args == (
        "POST",
        "https://api.attio.com/v2/objects/records/search",
    )
    assert request.call_args.kwargs["json"] == {
        "query": "Flower",
        "objects": ["companies"],
        "request_as": {"type": "workspace"},
        "limit": 25,
    }
    assert result == {"data": []}


def test_api_errors_are_secret_safe() -> None:
    """API errors should not expose tokens or response text."""
    with (
        patch(
            _HTTP_REQUEST,
            return_value=_response(
                {"message": "attio-secret"},
                status_code=401,
            ),
        ),
        pytest.raises(AttioApiError) as error,
    ):
        registry.invoke_connector(
            "attio_search_records",
            {"query": "Flower", "objects": ["companies"]},
            Mock(),
            _CREDENTIALS,
            {},
        )

    assert error.value.code == "http_error"
    assert "attio-secret" not in str(error.value)


def test_oauth_builds_url_and_exchanges_code() -> None:
    """OAuth should validate redirects and return Attio credentials."""
    url = _provider().build_authorization_url(
        redirect_uri=_REDIRECT_URI,
        state="oauth-state",
        pkce_challenge=None,
    )
    parsed = urlparse(url)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == (
        "https://app.attio.com/authorize"
    )
    assert parse_qs(parsed.query)["redirect_uri"] == [_REDIRECT_URI]
    with pytest.raises(ValueError):
        _provider().resolve_redirect_uri("https://attacker.example/callback")

    response = _response({"access_token": "attio-access"})
    with patch(_TOKEN_REQUEST, return_value=response) as post:
        credentials, config = _provider().exchange_code(
            code="authorization-code",
            redirect_uri=_REDIRECT_URI,
            pkce_verifier=None,
        )

    assert credentials == {"access_token": "attio-access"}
    assert config == {}
    assert post.call_args.kwargs["data"] == {
        "client_id": "client-id",
        "client_secret": "client-secret",
        "grant_type": "authorization_code",
        "code": "authorization-code",
        "redirect_uri": _REDIRECT_URI,
    }
    with pytest.raises(ValueError):
        _provider().exchange_code(
            code="authorization-code",
            redirect_uri="https://attacker.example/callback",
            pkce_verifier=None,
        )
