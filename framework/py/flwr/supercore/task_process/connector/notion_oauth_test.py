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


def _response(payload: object) -> Mock:
    response = Mock(status_code=200)
    response.json.return_value = payload
    return response


def test_authorization_url_and_redirect_allowlist() -> None:
    """Authorization should use Notion's documented public connection fields."""
    provider = _provider()
    authorization_url = provider.build_authorization_url(
        redirect_uri=_REDIRECT_URI,
        state="oauth-state",
        pkce_challenge="unused-pkce-challenge",
    )

    parsed = urlparse(authorization_url)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == (
        "https://api.notion.com/v1/oauth/authorize"
    )
    assert parse_qs(parsed.query) == {
        "client_id": ["client-id"],
        "redirect_uri": [_REDIRECT_URI],
        "response_type": ["code"],
        "owner": ["user"],
        "state": ["oauth-state"],
    }
    assert provider.resolve_redirect_uri(f" {_REDIRECT_URI} ") == _REDIRECT_URI
    with pytest.raises(ValueError):
        provider.resolve_redirect_uri("https://attacker.example/callback")


def test_exchange_code_separates_credentials_and_workspace_config() -> None:
    """OAuth exchange should store secrets separately from stable metadata."""
    response = _response(
        {
            "access_token": "ntn-access",
            "refresh_token": "ntn-refresh",
            "bot_id": "bot-123",
            "workspace_id": "workspace-123",
            "workspace_name": "Flower",
            "workspace_icon": "https://temporary.example/icon.png",
            "owner": {"user": {"id": "user-123", "name": "Ada"}},
        }
    )
    with patch(
        "flwr.supercore.task_process.connector.notion_oauth.requests.post",
        return_value=response,
    ) as post:
        credentials, config = _provider().exchange_code(
            code="authorization-code",
            redirect_uri=_REDIRECT_URI,
            pkce_verifier="unused-pkce-verifier",
        )

    assert post.call_args.args == ("https://api.notion.com/v1/oauth/token",)
    assert post.call_args.kwargs["auth"] == ("client-id", "client-secret")
    assert post.call_args.kwargs["headers"]["Notion-Version"] == "2026-03-11"
    assert post.call_args.kwargs["json"] == {
        "grant_type": "authorization_code",
        "code": "authorization-code",
        "redirect_uri": _REDIRECT_URI,
    }
    assert credentials == {
        "access_token": "ntn-access",
        "refresh_token": "ntn-refresh",
    }
    assert config == {
        "workspace_id": "workspace-123",
        "workspace_name": "Flower",
        "bot_id": "bot-123",
        "owner_user_id": "user-123",
    }


@pytest.mark.parametrize(
    ("response", "side_effect"),
    [
        (Mock(status_code=401), None),
        (_response({"error": "client-secret"}), None),
        (_response({}), None),
        (Mock(status_code=200, **{"json.side_effect": ValueError()}), None),
        (None, requests.RequestException("client-secret")),
    ],
)
def test_exchange_errors_do_not_expose_secrets(
    response: Mock | None, side_effect: Exception | None
) -> None:
    """OAuth failures must not expose codes, tokens, or client secrets."""
    with (
        patch(
            "flwr.supercore.task_process.connector.notion_oauth.requests.post",
            return_value=response,
            side_effect=side_effect,
        ),
        pytest.raises(NotionOAuthError) as error,
    ):
        _provider().exchange_code(
            code="authorization-code",
            redirect_uri=_REDIRECT_URI,
            pkce_verifier=None,
        )

    assert all(
        secret not in str(error.value)
        for secret in ("authorization-code", "client-secret", "ntn-access")
    )


def test_provider_registration_requires_complete_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Notion should register only with a complete OAuth environment."""
    for name in (
        NOTION_CLIENT_ID_ENV,
        NOTION_CLIENT_SECRET_ENV,
        NOTION_REDIRECT_URI_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    assert not get_configured_connector_oauth_providers()

    monkeypatch.setenv(NOTION_CLIENT_ID_ENV, "client-id")
    monkeypatch.setenv(NOTION_CLIENT_SECRET_ENV, "client-secret")
    monkeypatch.setenv(NOTION_REDIRECT_URI_ENV, _REDIRECT_URI)
    providers = get_configured_connector_oauth_providers()
    assert len(providers) == 1
    assert providers[0].connector_ref == "notion"

    monkeypatch.delenv(NOTION_CLIENT_SECRET_ENV)
    monkeypatch.delenv(NOTION_REDIRECT_URI_ENV)
    with pytest.raises(RuntimeError) as error:
        get_configured_connector_oauth_providers()
    assert NOTION_CLIENT_SECRET_ENV in str(error.value)
    assert NOTION_REDIRECT_URI_ENV in str(error.value)
    assert "client-id" not in str(error.value)
