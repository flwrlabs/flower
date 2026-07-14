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
"""Tests for the Slack OAuth provider."""

from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

import pytest
import requests

from .slack_oauth import (
    SLACK_CLIENT_ID_ENV,
    SLACK_CLIENT_SECRET_ENV,
    SLACK_REDIRECT_URI_ENV,
    SLACK_USER_SCOPES,
    SlackOAuthError,
    SlackOAuthProvider,
    get_configured_connector_oauth_providers,
)

_REDIRECT_URI = "https://client.example/oauth/slack"


def _provider() -> SlackOAuthProvider:
    return SlackOAuthProvider(
        client_id="client-id",
        client_secret="client-secret",
        redirect_uri=_REDIRECT_URI,
    )


def test_build_authorization_url_requests_read_only_user_scopes() -> None:
    """Slack authorization should request the approved user-token scopes."""
    provider = _provider()

    authorization_url = provider.build_authorization_url(
        redirect_uri=_REDIRECT_URI,
        state="oauth-state",
        pkce_challenge=None,
    )

    parsed = urlparse(authorization_url)
    params = parse_qs(parsed.query)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == (
        "https://slack.com/oauth/v2/authorize"
    )
    assert params == {
        "client_id": ["client-id"],
        "redirect_uri": [_REDIRECT_URI],
        "state": ["oauth-state"],
        "user_scope": [",".join(SLACK_USER_SCOPES)],
    }
    assert all(not scope.endswith(":write") for scope in SLACK_USER_SCOPES)


def test_resolve_redirect_uri_requires_configured_value() -> None:
    """Slack should reject redirect URIs not configured for the application."""
    provider = _provider()

    assert provider.resolve_redirect_uri(f" {_REDIRECT_URI} ") == _REDIRECT_URI
    with pytest.raises(ValueError):
        provider.resolve_redirect_uri("https://attacker.example/callback")


def test_exchange_code_returns_user_credentials_and_workspace_config() -> None:
    """Slack OAuth responses should be separated into secrets and metadata."""
    response = Mock(status_code=200)
    response.json.return_value = {
        "ok": True,
        "team": {"id": "T123", "name": "Flower"},
        "enterprise": {"id": "E123"},
        "authed_user": {
            "id": "U123",
            "scope": "search:read,channels:read",
            "access_token": "xoxp-access",
            "refresh_token": "xoxe-refresh",
            "expires_in": 43200,
        },
    }

    with patch(
        "flwr.supercore.task_process.connector.slack_oauth.requests.post",
        return_value=response,
    ) as post:
        result = _provider().exchange_code(
            code="authorization-code",
            redirect_uri=_REDIRECT_URI,
            pkce_verifier=None,
        )

    post.assert_called_once_with(
        "https://slack.com/api/oauth.v2.access",
        auth=("client-id", "client-secret"),
        data={"code": "authorization-code", "redirect_uri": _REDIRECT_URI},
        timeout=30.0,
    )
    credentials, config = result
    assert credentials == {
        "access_token": "xoxp-access",
        "refresh_token": "xoxe-refresh",
        "expires_in": 43200,
    }
    assert config == {
        "user_id": "U123",
        "scopes": ["search:read", "channels:read"],
        "team_id": "T123",
        "team_name": "Flower",
        "enterprise_id": "E123",
    }


@pytest.mark.parametrize(
    "response",
    [
        Mock(status_code=401),
        Mock(status_code=200, **{"json.return_value": {"ok": False, "error": "bad"}}),
        Mock(status_code=200, **{"json.return_value": {"ok": True}}),
    ],
)
def test_exchange_code_errors_do_not_expose_secrets(response: Mock) -> None:
    """OAuth failures must not expose codes, tokens, or client secrets."""
    secret_values = ["authorization-code", "client-secret", "xoxp-secret"]
    if response.status_code == 200 and response.json.return_value.get("ok") is False:
        response.json.return_value["error"] = "xoxp-secret"

    with (
        patch(
            "flwr.supercore.task_process.connector.slack_oauth.requests.post",
            return_value=response,
        ),
        pytest.raises(SlackOAuthError) as error,
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
            "flwr.supercore.task_process.connector.slack_oauth.requests.post",
            side_effect=requests.RequestException("client-secret"),
        ),
        pytest.raises(SlackOAuthError, match="Slack OAuth exchange failed"),
    ):
        _provider().exchange_code(
            code="authorization-code",
            redirect_uri=_REDIRECT_URI,
            pkce_verifier=None,
        )


def test_configured_provider_uses_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A complete environment configuration should register Slack."""
    monkeypatch.setenv(SLACK_CLIENT_ID_ENV, "client-id")
    monkeypatch.setenv(SLACK_CLIENT_SECRET_ENV, "client-secret")
    monkeypatch.setenv(SLACK_REDIRECT_URI_ENV, _REDIRECT_URI)

    providers = get_configured_connector_oauth_providers()

    assert len(providers) == 1
    assert providers[0].connector_ref == "slack"


def test_provider_is_not_registered_without_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slack should stay unavailable when no OAuth application is configured."""
    monkeypatch.delenv(SLACK_CLIENT_ID_ENV, raising=False)
    monkeypatch.delenv(SLACK_CLIENT_SECRET_ENV, raising=False)
    monkeypatch.delenv(SLACK_REDIRECT_URI_ENV, raising=False)

    assert get_configured_connector_oauth_providers() == []


def test_partial_environment_configuration_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partially configured Slack OAuth should fail at startup."""
    monkeypatch.setenv(SLACK_CLIENT_ID_ENV, "client-id")
    monkeypatch.delenv(SLACK_CLIENT_SECRET_ENV, raising=False)
    monkeypatch.delenv(SLACK_REDIRECT_URI_ENV, raising=False)

    with pytest.raises(RuntimeError) as error:
        get_configured_connector_oauth_providers()

    assert SLACK_CLIENT_SECRET_ENV in str(error.value)
    assert SLACK_REDIRECT_URI_ENV in str(error.value)
    assert "client-id" not in str(error.value)
