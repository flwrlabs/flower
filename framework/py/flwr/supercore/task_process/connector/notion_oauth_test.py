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
"""Tests for the Notion OAuth provider."""

from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

import pytest
import requests

from .notion_oauth import (
    NOTION_CLIENT_ID_ENV,
    NOTION_CLIENT_SECRET_ENV,
    NOTION_REDIRECT_URI_ENV,
    NotionOAuthError,
    NotionOAuthProvider,
    get_configured_connector_oauth_providers,
)

_REDIRECT_URI = "https://client.example/oauth/notion"


def _provider() -> NotionOAuthProvider:
    return NotionOAuthProvider(
        client_id="client-id",
        client_secret="client-secret",
        redirect_uri=_REDIRECT_URI,
    )


def test_build_authorization_url_uses_notion_public_connection_flow() -> None:
    """Authorization should use Notion's documented public connection fields."""
    authorization_url = _provider().build_authorization_url(
        redirect_uri=_REDIRECT_URI,
        state="oauth-state",
        pkce_challenge="unused-pkce-challenge",
    )

    parsed = urlparse(authorization_url)
    params = parse_qs(parsed.query)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == (
        "https://api.notion.com/v1/oauth/authorize"
    )
    assert params == {
        "client_id": ["client-id"],
        "redirect_uri": [_REDIRECT_URI],
        "response_type": ["code"],
        "owner": ["user"],
        "state": ["oauth-state"],
    }


def test_resolve_redirect_uri_requires_configured_value() -> None:
    """Notion should reject redirect URIs not configured for the connection."""
    provider = _provider()

    assert provider.resolve_redirect_uri(f" {_REDIRECT_URI} ") == _REDIRECT_URI
    with pytest.raises(ValueError):
        provider.resolve_redirect_uri("https://attacker.example/callback")


def test_exchange_code_returns_credentials_and_stable_workspace_config() -> None:
    """Notion OAuth responses should separate secrets from stable metadata."""
    response = Mock(status_code=200)
    response.json.return_value = {
        "access_token": "ntn-access",
        "refresh_token": "ntn-refresh",
        "expires_in": 3600,
        "bot_id": "bot-123",
        "workspace_id": "workspace-123",
        "workspace_name": "Flower",
        "workspace_icon": "https://temporary.example/icon.png",
        "duplicated_template_id": None,
        "owner": {
            "type": "user",
            "user": {
                "id": "user-123",
                "name": "Ada",
                "person": {"email": "ada@example.com"},
            },
        },
    }

    with patch(
        "flwr.supercore.task_process.connector.notion_oauth.requests.post",
        return_value=response,
    ) as post:
        result = _provider().exchange_code(
            code="authorization-code",
            redirect_uri=_REDIRECT_URI,
            pkce_verifier="unused-pkce-verifier",
        )

    post.assert_called_once_with(
        "https://api.notion.com/v1/oauth/token",
        auth=("client-id", "client-secret"),
        headers={
            "Accept": "application/json",
            "Notion-Version": "2026-03-11",
        },
        json={
            "grant_type": "authorization_code",
            "code": "authorization-code",
            "redirect_uri": _REDIRECT_URI,
        },
        timeout=30.0,
    )
    credentials, config = result
    assert credentials == {
        "access_token": "ntn-access",
        "refresh_token": "ntn-refresh",
        "expires_in": 3600,
    }
    assert config == {
        "workspace_id": "workspace-123",
        "workspace_name": "Flower",
        "bot_id": "bot-123",
        "owner_user_id": "user-123",
    }


def test_exchange_code_accepts_nullable_refresh_token() -> None:
    """Notion's nullable refresh token should not invalidate an access token."""
    response = Mock(status_code=200)
    response.json.return_value = {
        "access_token": "ntn-access",
        "refresh_token": None,
    }
    with patch(
        "flwr.supercore.task_process.connector.notion_oauth.requests.post",
        return_value=response,
    ):
        credentials, config = _provider().exchange_code(
            code="authorization-code",
            redirect_uri=_REDIRECT_URI,
            pkce_verifier=None,
        )

    assert credentials == {"access_token": "ntn-access"}
    assert not config


@pytest.mark.parametrize(
    "response",
    [
        Mock(status_code=401),
        Mock(status_code=200, **{"json.return_value": {"error": "client-secret"}}),
        Mock(status_code=200, **{"json.return_value": {}}),
        Mock(status_code=200, **{"json.side_effect": ValueError()}),
    ],
)
def test_exchange_code_errors_do_not_expose_secrets(response: Mock) -> None:
    """OAuth failures must not expose codes, tokens, or client secrets."""
    secret_values = ["authorization-code", "client-secret", "ntn-access"]
    with (
        patch(
            "flwr.supercore.task_process.connector.notion_oauth.requests.post",
            return_value=response,
        ),
        pytest.raises(NotionOAuthError) as error,
    ):
        _provider().exchange_code(
            code="authorization-code",
            redirect_uri=_REDIRECT_URI,
            pkce_verifier=None,
        )

    assert all(value not in str(error.value) for value in secret_values)


def test_exchange_code_translates_request_failures() -> None:
    """Transport error details should stay behind the OAuth provider boundary."""
    with (
        patch(
            "flwr.supercore.task_process.connector.notion_oauth.requests.post",
            side_effect=requests.RequestException("client-secret"),
        ),
        pytest.raises(NotionOAuthError, match="Notion OAuth exchange failed"),
    ):
        _provider().exchange_code(
            code="authorization-code",
            redirect_uri=_REDIRECT_URI,
            pkce_verifier=None,
        )


def test_configured_provider_uses_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A complete environment configuration should register Notion."""
    monkeypatch.setenv(NOTION_CLIENT_ID_ENV, "client-id")
    monkeypatch.setenv(NOTION_CLIENT_SECRET_ENV, "client-secret")
    monkeypatch.setenv(NOTION_REDIRECT_URI_ENV, _REDIRECT_URI)

    providers = get_configured_connector_oauth_providers()

    assert len(providers) == 1
    assert providers[0].connector_ref == "notion"


def test_provider_is_not_registered_without_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Notion should stay unavailable when no OAuth connection is configured."""
    monkeypatch.delenv(NOTION_CLIENT_ID_ENV, raising=False)
    monkeypatch.delenv(NOTION_CLIENT_SECRET_ENV, raising=False)
    monkeypatch.delenv(NOTION_REDIRECT_URI_ENV, raising=False)

    assert not get_configured_connector_oauth_providers()


def test_partial_environment_configuration_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partially configured Notion OAuth should fail at startup."""
    monkeypatch.setenv(NOTION_CLIENT_ID_ENV, "client-id")
    monkeypatch.delenv(NOTION_CLIENT_SECRET_ENV, raising=False)
    monkeypatch.delenv(NOTION_REDIRECT_URI_ENV, raising=False)

    with pytest.raises(RuntimeError) as error:
        get_configured_connector_oauth_providers()

    assert NOTION_CLIENT_SECRET_ENV in str(error.value)
    assert NOTION_REDIRECT_URI_ENV in str(error.value)
    assert "client-id" not in str(error.value)
