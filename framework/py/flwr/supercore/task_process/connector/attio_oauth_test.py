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
"""Tests for the Attio OAuth provider."""

from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

import pytest
import requests

from .attio_oauth import AttioOAuthError, AttioOAuthProvider

_REDIRECT_URI = "https://client.example/oauth/attio"


def _provider() -> AttioOAuthProvider:
    return AttioOAuthProvider(
        client_id="client-id",
        client_secret="client-secret",
        redirect_uri=_REDIRECT_URI,
    )


def _response(payload: object, *, status_code: int = 200) -> Mock:
    response = Mock(status_code=status_code)
    response.json.return_value = payload
    return response


def test_authorization_url_validates_redirect_uri() -> None:
    """Authorization should use the configured Attio redirect URI."""
    url = _provider().build_authorization_url(
        redirect_uri=_REDIRECT_URI,
        state="oauth-state",
        pkce_challenge=None,
    )
    parsed = urlparse(url)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == (
        "https://app.attio.com/authorize"
    )
    assert parse_qs(parsed.query) == {
        "response_type": ["code"],
        "client_id": ["client-id"],
        "redirect_uri": [_REDIRECT_URI],
        "state": ["oauth-state"],
    }
    with pytest.raises(ValueError):
        _provider().resolve_redirect_uri("https://attacker.example/callback")


def test_exchange_returns_credentials_and_workspace_config() -> None:
    """OAuth exchange should separate the token from public workspace metadata."""
    with patch(
        "flwr.supercore.task_process.connector.attio_oauth.requests.post",
        return_value=_response(
            {"access_token": "attio-access", "token_type": "Bearer"}
        ),
    ) as post:
        credentials, config = _provider().exchange_code(
            code="authorization-code",
            redirect_uri=_REDIRECT_URI,
            pkce_verifier=None,
        )

    assert credentials == {"access_token": "attio-access"}
    assert config == {}
    post.assert_called_once()
    assert post.call_args.kwargs["data"] == {
        "client_id": "client-id",
        "client_secret": "client-secret",
        "grant_type": "authorization_code",
        "code": "authorization-code",
        "redirect_uri": _REDIRECT_URI,
    }


def test_exchange_transport_errors_are_secret_safe() -> None:
    """OAuth transport failures should not expose credentials."""
    with (
        patch(
            "flwr.supercore.task_process.connector.attio_oauth.requests.post",
            side_effect=requests.RequestException("client-secret"),
        ),
        pytest.raises(AttioOAuthError) as error,
    ):
        _provider().exchange_code(
            code="authorization-code",
            redirect_uri=_REDIRECT_URI,
            pkce_verifier=None,
        )

    assert "client-secret" not in str(error.value)
