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
"""Tests for the GitHub OAuth provider."""

from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

import pytest
import requests

from .github import GITHUB_API_VERSION
from .github_oauth import (
    GITHUB_CLIENT_ID_ENV,
    GITHUB_CLIENT_SECRET_ENV,
    GITHUB_REDIRECT_URI_ENV,
    GitHubOAuthError,
    GitHubOAuthProvider,
    get_configured_connector_oauth_providers,
)

_REDIRECT_URI = "https://client.example/oauth/github"


def _provider() -> GitHubOAuthProvider:
    return GitHubOAuthProvider(
        client_id="client-id",
        client_secret="client-secret",
        redirect_uri=_REDIRECT_URI,
    )


def _response(payload: object, *, status_code: int = 200) -> Mock:
    response = Mock(status_code=status_code)
    response.json.return_value = payload
    return response


def test_authorization_url_requests_public_access_with_pkce() -> None:
    """Authorization should request no scope and bind the flow with PKCE."""
    provider = _provider()
    authorization_url = provider.build_authorization_url(
        redirect_uri=_REDIRECT_URI,
        state="oauth-state",
        pkce_challenge="pkce-challenge",
    )

    parsed = urlparse(authorization_url)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == (
        "https://github.com/login/oauth/authorize"
    )
    params = parse_qs(parsed.query)
    assert params == {
        "client_id": ["client-id"],
        "redirect_uri": [_REDIRECT_URI],
        "state": ["oauth-state"],
        "code_challenge": ["pkce-challenge"],
        "code_challenge_method": ["S256"],
    }
    assert "scope" not in params
    assert provider.resolve_redirect_uri(f" {_REDIRECT_URI} ") == _REDIRECT_URI
    with pytest.raises(ValueError):
        provider.resolve_redirect_uri("https://attacker.example/callback")


def test_exchange_code_verifies_account_and_separates_credentials() -> None:
    """OAuth exchange should store only stable public account metadata as config."""
    token_response = _response(
        {
            "access_token": "gho-access",
            "scope": "",
            "token_type": "bearer",
        }
    )
    account_response = _response(
        {
            "id": 123,
            "login": "octocat",
            "avatar_url": "https://avatars.githubusercontent.com/u/123",
        }
    )
    with (
        patch(
            "flwr.supercore.task_process.connector.github_oauth.requests.post",
            return_value=token_response,
        ) as post,
        patch(
            "flwr.supercore.task_process.connector.github_oauth.requests.get",
            return_value=account_response,
        ) as get,
    ):
        credentials, config = _provider().exchange_code(
            code="authorization-code",
            redirect_uri=_REDIRECT_URI,
            pkce_verifier="pkce-verifier",
        )

    post.assert_called_once_with(
        "https://github.com/login/oauth/access_token",
        headers={"Accept": "application/json"},
        data={
            "client_id": "client-id",
            "client_secret": "client-secret",
            "code": "authorization-code",
            "redirect_uri": _REDIRECT_URI,
            "code_verifier": "pkce-verifier",
        },
        timeout=30.0,
    )
    get.assert_called_once_with(
        "https://api.github.com/user",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer gho-access",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        },
        timeout=30.0,
    )
    assert credentials == {"access_token": "gho-access"}
    assert config == {
        "account_id": 123,
        "login": "octocat",
        "scopes": [],
        "token_type": "bearer",
    }


def test_exchange_rejects_tokens_with_repository_scopes() -> None:
    """A previous broad app grant must not turn this into private/write access."""
    with (
        patch(
            "flwr.supercore.task_process.connector.github_oauth.requests.post",
            return_value=_response(
                {
                    "access_token": "gho-access",
                    "scope": "repo, user",
                    "token_type": "bearer",
                }
            ),
        ),
        patch("flwr.supercore.task_process.connector.github_oauth.requests.get") as get,
        pytest.raises(GitHubOAuthError, match="unsupported scopes"),
    ):
        _provider().exchange_code(
            code="authorization-code",
            redirect_uri=_REDIRECT_URI,
            pkce_verifier="pkce-verifier",
        )

    get.assert_not_called()


def test_exchange_errors_do_not_expose_secrets() -> None:
    """Token failures must not expose codes, tokens, or client secrets."""
    with (
        patch(
            "flwr.supercore.task_process.connector.github_oauth.requests.post",
            side_effect=requests.RequestException("client-secret"),
        ),
        pytest.raises(GitHubOAuthError) as error,
    ):
        _provider().exchange_code(
            code="authorization-code",
            redirect_uri=_REDIRECT_URI,
            pkce_verifier="pkce-verifier",
        )

    assert all(
        secret not in str(error.value)
        for secret in (
            "authorization-code",
            "client-secret",
            "pkce-verifier",
        )
    )


def test_account_lookup_errors_do_not_expose_token() -> None:
    """Account verification failures must remain secret-safe."""
    with (
        patch(
            "flwr.supercore.task_process.connector.github_oauth.requests.post",
            return_value=_response(
                {
                    "access_token": "gho-access",
                    "scope": "",
                    "token_type": "bearer",
                }
            ),
        ),
        patch(
            "flwr.supercore.task_process.connector.github_oauth.requests.get",
            side_effect=requests.RequestException("gho-access"),
        ),
        pytest.raises(GitHubOAuthError) as error,
    ):
        _provider().exchange_code(
            code="authorization-code",
            redirect_uri=_REDIRECT_URI,
            pkce_verifier="pkce-verifier",
        )

    assert "gho-access" not in str(error.value)


def test_provider_registration_requires_complete_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GitHub should register only with a complete OAuth environment."""
    for name in (
        GITHUB_CLIENT_ID_ENV,
        GITHUB_CLIENT_SECRET_ENV,
        GITHUB_REDIRECT_URI_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    assert not get_configured_connector_oauth_providers()

    monkeypatch.setenv(GITHUB_CLIENT_ID_ENV, "client-id")
    monkeypatch.setenv(GITHUB_CLIENT_SECRET_ENV, "client-secret")
    monkeypatch.setenv(GITHUB_REDIRECT_URI_ENV, _REDIRECT_URI)
    providers = get_configured_connector_oauth_providers()
    assert len(providers) == 1
    assert providers[0].connector_ref == "github"

    monkeypatch.delenv(GITHUB_CLIENT_SECRET_ENV)
    monkeypatch.delenv(GITHUB_REDIRECT_URI_ENV)
    with pytest.raises(RuntimeError) as error:
        get_configured_connector_oauth_providers()
    assert GITHUB_CLIENT_SECRET_ENV in str(error.value)
    assert GITHUB_REDIRECT_URI_ENV in str(error.value)
    assert "client-id" not in str(error.value)
