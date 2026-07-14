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
"""Slack OAuth provider."""

import os
from urllib.parse import urlencode

import requests

from flwr.supercore.typing import JSONObject

from .oauth import OAuthConnectorProvider
from .slack import SLACK_CONNECTOR_REF

SLACK_CLIENT_ID_ENV = "FLWR_SLACK_CLIENT_ID"
SLACK_CLIENT_SECRET_ENV = "FLWR_SLACK_CLIENT_SECRET"
SLACK_REDIRECT_URI_ENV = "FLWR_SLACK_REDIRECT_URI"

_SLACK_AUTHORIZE_URL = "https://slack.com/oauth/v2/authorize"
_SLACK_TOKEN_URL = "https://slack.com/api/oauth.v2.access"
_REQUEST_TIMEOUT = 30.0

# `search.messages` requires the legacy user-token `search:read` scope. The
# remaining scopes grant read-only access to Slack conversations.
SLACK_USER_SCOPES = (
    "search:read",
    "channels:read",
    "groups:read",
    "im:read",
    "mpim:read",
    "channels:history",
    "groups:history",
    "im:history",
    "mpim:history",
)


class SlackOAuthError(RuntimeError):
    """Secret-safe Slack OAuth failure."""


class SlackOAuthProvider:
    """Slack implementation of the OAuth provider contract."""

    connector_ref = SLACK_CONNECTOR_REF
    display_name = "Slack"
    description = "Search and read messages, conversations, and threads."

    def __init__(
        self, *, client_id: str, client_secret: str, redirect_uri: str
    ) -> None:
        client_id = client_id.strip()
        client_secret = client_secret.strip()
        redirect_uri = redirect_uri.strip()
        if not client_id or not client_secret or not redirect_uri:
            raise ValueError("Slack OAuth configuration is incomplete.")
        self._client_id = client_id
        self._client_secret = client_secret
        self._redirect_uri = redirect_uri

    def resolve_redirect_uri(self, requested_redirect_uri: str) -> str:
        """Require the redirect URI configured for the Slack application."""
        if requested_redirect_uri.strip() != self._redirect_uri:
            raise ValueError("Slack redirect URI is not allowed.")
        return self._redirect_uri

    def build_authorization_url(
        self,
        *,
        redirect_uri: str,
        state: str,
        pkce_challenge: str | None,
    ) -> str:
        """Build a Slack user-token authorization URL."""
        if pkce_challenge is not None:
            raise ValueError("Slack PKCE is not enabled for this provider.")
        params = {
            "client_id": self._client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "user_scope": ",".join(SLACK_USER_SCOPES),
        }
        return f"{_SLACK_AUTHORIZE_URL}?{urlencode(params)}"

    def exchange_code(
        self,
        *,
        code: str,
        redirect_uri: str,
        pkce_verifier: str | None,
    ) -> tuple[JSONObject, JSONObject]:
        """Exchange a Slack authorization code for user-token credentials."""
        data = {"code": code, "redirect_uri": redirect_uri}
        if pkce_verifier is not None:
            data["code_verifier"] = pkce_verifier
        try:
            response = requests.post(
                _SLACK_TOKEN_URL,
                auth=(self._client_id, self._client_secret),
                data=data,
                timeout=_REQUEST_TIMEOUT,
            )
        except requests.RequestException:
            raise SlackOAuthError("Slack OAuth exchange failed.") from None
        if response.status_code >= 400:
            raise SlackOAuthError("Slack OAuth exchange failed.")

        try:
            payload = response.json()
        except ValueError:
            raise SlackOAuthError("Slack OAuth returned an invalid response.") from None
        if not isinstance(payload, dict):
            raise SlackOAuthError("Slack OAuth returned an invalid response.")
        if payload.get("ok") is not True:
            raise SlackOAuthError("Slack OAuth exchange failed.")

        authed_user = payload.get("authed_user")
        if not isinstance(authed_user, dict):
            raise SlackOAuthError("Slack OAuth response has no authorized user.")
        access_token = authed_user.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise SlackOAuthError("Slack OAuth response has no user access token.")

        credentials: JSONObject = {"access_token": access_token}
        refresh_token = authed_user.get("refresh_token")
        if isinstance(refresh_token, str) and refresh_token:
            credentials["refresh_token"] = refresh_token
        expires_in = authed_user.get("expires_in")
        if isinstance(expires_in, int) and not isinstance(expires_in, bool):
            credentials["expires_in"] = expires_in

        config: JSONObject = {
            "user_id": _optional_string(authed_user.get("id")),
            "scopes": _parse_scopes(authed_user.get("scope")),
        }
        team = payload.get("team")
        if isinstance(team, dict):
            config["team_id"] = _optional_string(team.get("id"))
            config["team_name"] = _optional_string(team.get("name"))
        enterprise = payload.get("enterprise")
        if isinstance(enterprise, dict):
            config["enterprise_id"] = _optional_string(enterprise.get("id"))

        return credentials, config


def get_configured_connector_oauth_providers() -> list[OAuthConnectorProvider]:
    """Return configured built-in OAuth connector providers."""
    values = {
        SLACK_CLIENT_ID_ENV: os.getenv(SLACK_CLIENT_ID_ENV, "").strip(),
        SLACK_CLIENT_SECRET_ENV: os.getenv(SLACK_CLIENT_SECRET_ENV, "").strip(),
        SLACK_REDIRECT_URI_ENV: os.getenv(SLACK_REDIRECT_URI_ENV, "").strip(),
    }
    if not any(values.values()):
        return []
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError(
            "Slack OAuth configuration requires environment variables: "
            + ", ".join(missing)
            + "."
        )
    return [
        SlackOAuthProvider(
            client_id=values[SLACK_CLIENT_ID_ENV],
            client_secret=values[SLACK_CLIENT_SECRET_ENV],
            redirect_uri=values[SLACK_REDIRECT_URI_ENV],
        )
    ]


def _optional_string(value: object) -> str | None:
    """Return a non-empty string or None."""
    return value if isinstance(value, str) and value else None


def _parse_scopes(value: object) -> list[str]:
    """Parse Slack's comma-delimited granted user scopes."""
    if not isinstance(value, str):
        return []
    return [scope.strip() for scope in value.split(",") if scope.strip()]
