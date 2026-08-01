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
"""GitHub OAuth provider for public repository access."""

import os
from urllib.parse import urlencode

import requests

from flwr.supercore.typing import JSONObject

from .github import GITHUB_API_VERSION, GITHUB_CONNECTOR_REF
from .oauth import OAuthConnectorProvider

GITHUB_CLIENT_ID_ENV = "FLWR_GITHUB_CLIENT_ID"
GITHUB_CLIENT_SECRET_ENV = "FLWR_GITHUB_CLIENT_SECRET"
GITHUB_REDIRECT_URI_ENV = "FLWR_GITHUB_REDIRECT_URI"

_GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
_GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
_GITHUB_USER_URL = "https://api.github.com/user"
_REQUEST_TIMEOUT = 30.0


class GitHubOAuthError(RuntimeError):
    """Secret-safe GitHub OAuth failure."""


class GitHubOAuthProvider:
    """GitHub implementation of the OAuth provider contract."""

    connector_ref = GITHUB_CONNECTOR_REF
    display_name = "GitHub"
    description = "Search code and read files in public repositories."

    def __init__(
        self, *, client_id: str, client_secret: str, redirect_uri: str
    ) -> None:
        client_id = client_id.strip()
        client_secret = client_secret.strip()
        redirect_uri = redirect_uri.strip()
        if not client_id or not client_secret or not redirect_uri:
            raise ValueError("GitHub OAuth configuration is incomplete.")
        self._client_id = client_id
        self._client_secret = client_secret
        self._redirect_uri = redirect_uri

    def resolve_redirect_uri(self, requested_redirect_uri: str) -> str:
        """Require the redirect URI configured for the GitHub application."""
        if requested_redirect_uri.strip() != self._redirect_uri:
            raise ValueError("GitHub redirect URI is not allowed.")
        return self._redirect_uri

    def build_authorization_url(
        self,
        *,
        redirect_uri: str,
        state: str,
        pkce_challenge: str | None,
    ) -> str:
        """Build a public-information GitHub authorization URL."""
        params = {
            "client_id": self._client_id,
            "redirect_uri": redirect_uri,
            "state": state,
        }
        if pkce_challenge is not None:
            params["code_challenge"] = pkce_challenge
            params["code_challenge_method"] = "S256"
        return f"{_GITHUB_AUTHORIZE_URL}?{urlencode(params)}"

    def exchange_code(
        self,
        *,
        code: str,
        redirect_uri: str,
        pkce_verifier: str | None,
    ) -> tuple[JSONObject, JSONObject]:
        """Exchange a GitHub authorization code and verify its account."""
        if not code:
            raise GitHubOAuthError("GitHub OAuth exchange failed.")
        data = {
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "code": code,
            "redirect_uri": redirect_uri,
        }
        if pkce_verifier is not None:
            data["code_verifier"] = pkce_verifier
        try:
            response = requests.post(
                _GITHUB_TOKEN_URL,
                headers={"Accept": "application/json"},
                data=data,
                timeout=_REQUEST_TIMEOUT,
            )
        except requests.RequestException:
            raise GitHubOAuthError("GitHub OAuth exchange failed.") from None
        if response.status_code >= 400:
            raise GitHubOAuthError("GitHub OAuth exchange failed.")
        payload = _response_object(
            response, "GitHub OAuth returned an invalid response."
        )
        if "error" in payload:
            raise GitHubOAuthError("GitHub OAuth exchange failed.")

        access_token = _required_string(payload, "access_token")
        token_type = _required_string(payload, "token_type").lower()
        if token_type != "bearer":
            raise GitHubOAuthError("GitHub OAuth returned an unsupported token type.")
        scopes = _parse_scopes(payload.get("scope"))
        if scopes:
            raise GitHubOAuthError("GitHub OAuth granted unsupported scopes.")
        account = _get_authenticated_account(access_token)
        account_id = account.get("id")
        if isinstance(account_id, bool) or not isinstance(account_id, int):
            raise GitHubOAuthError(
                "GitHub account lookup returned an invalid response."
            )
        login = _required_string(
            account,
            "login",
            error="GitHub account lookup returned an invalid response.",
        )
        return {"access_token": access_token}, {
            "account_id": account_id,
            "login": login,
            "scopes": scopes,
            "token_type": token_type,
        }


def get_configured_connector_oauth_providers() -> list[OAuthConnectorProvider]:
    """Return the configured GitHub OAuth provider, if available."""
    values = {
        GITHUB_CLIENT_ID_ENV: os.getenv(GITHUB_CLIENT_ID_ENV, "").strip(),
        GITHUB_CLIENT_SECRET_ENV: os.getenv(GITHUB_CLIENT_SECRET_ENV, "").strip(),
        GITHUB_REDIRECT_URI_ENV: os.getenv(GITHUB_REDIRECT_URI_ENV, "").strip(),
    }
    if not any(values.values()):
        return []
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError(
            "GitHub OAuth configuration requires environment variables: "
            + ", ".join(missing)
            + "."
        )
    return [
        GitHubOAuthProvider(
            client_id=values[GITHUB_CLIENT_ID_ENV],
            client_secret=values[GITHUB_CLIENT_SECRET_ENV],
            redirect_uri=values[GITHUB_REDIRECT_URI_ENV],
        )
    ]


def _get_authenticated_account(access_token: str) -> JSONObject:
    """Verify the token by loading its stable GitHub account identity."""
    try:
        response = requests.get(
            _GITHUB_USER_URL,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {access_token}",
                "X-GitHub-Api-Version": GITHUB_API_VERSION,
            },
            timeout=_REQUEST_TIMEOUT,
        )
    except requests.RequestException:
        raise GitHubOAuthError("GitHub account lookup failed.") from None
    if response.status_code >= 400:
        raise GitHubOAuthError("GitHub account lookup failed.")
    return _response_object(
        response,
        "GitHub account lookup returned an invalid response.",
    )


def _response_object(response: requests.Response, error: str) -> JSONObject:
    """Decode one JSON object without exposing response details."""
    try:
        payload = response.json()
    except ValueError:
        raise GitHubOAuthError(error) from None
    if not isinstance(payload, dict):
        raise GitHubOAuthError(error)
    return payload


def _required_string(
    payload: JSONObject,
    key: str,
    *,
    error: str = "GitHub OAuth response is missing credentials.",
) -> str:
    """Read a required string without including its value in errors."""
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise GitHubOAuthError(error)
    return value


def _parse_scopes(value: object) -> list[str]:
    """Parse GitHub's comma-delimited granted scopes."""
    if not isinstance(value, str):
        return []
    return [scope.strip() for scope in value.split(",") if scope.strip()]
