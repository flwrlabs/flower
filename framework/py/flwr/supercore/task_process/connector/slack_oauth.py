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
from collections.abc import Mapping

import requests

from flwr.supercore.typing import JSONObject

from .json_utils import object_field, required_string_field, string_field
from .oauth import BaseOAuthProvider, OAuthConnectorProvider, load_oauth_provider
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


class SlackOAuthProvider(BaseOAuthProvider):
    """Slack implementation of the OAuth provider contract."""

    connector_ref = SLACK_CONNECTOR_REF
    display_name = "Slack"
    description = "Search and read messages, conversations, and threads."
    authorize_url = _SLACK_AUTHORIZE_URL
    error_type = SlackOAuthError

    def authorization_parameters(
        self,
        *,
        redirect_uri: str,
        state: str,
        pkce_challenge: str | None,
    ) -> Mapping[str, str]:
        """Return Slack user-token authorization parameters."""
        if pkce_challenge is not None:
            raise ValueError("Slack PKCE is not enabled for this provider.")
        return {
            "client_id": self._client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "user_scope": ",".join(SLACK_USER_SCOPES),
        }

    def request_token(
        self,
        *,
        code: str,
        redirect_uri: str,
        pkce_verifier: str | None,
    ) -> requests.Response:
        """Exchange a Slack authorization code for a token response."""
        data = {"code": code, "redirect_uri": redirect_uri}
        if pkce_verifier is not None:
            data["code_verifier"] = pkce_verifier
        return requests.post(
            _SLACK_TOKEN_URL,
            auth=(self._client_id, self._client_secret),
            data=data,
            timeout=_REQUEST_TIMEOUT,
        )

    def parse_token_response(
        self, payload: JSONObject
    ) -> tuple[JSONObject, JSONObject]:
        """Extract Slack user credentials and workspace configuration."""
        if payload.get("ok") is not True:
            raise SlackOAuthError("Slack OAuth exchange failed.")

        authed_user = object_field(payload, "authed_user", error=self._error)
        access_token = required_string_field(
            authed_user, "access_token", error=self._error
        )

        credentials: JSONObject = {"access_token": access_token}
        refresh_token = authed_user.get("refresh_token")
        if isinstance(refresh_token, str) and refresh_token:
            credentials["refresh_token"] = refresh_token
        expires_in = authed_user.get("expires_in")
        if isinstance(expires_in, int) and not isinstance(expires_in, bool):
            credentials["expires_in"] = expires_in

        config: JSONObject = {
            "user_id": string_field(authed_user, "id") or None,
            "scopes": _parse_scopes(authed_user.get("scope")),
        }
        team = payload.get("team")
        if isinstance(team, dict):
            config["team_id"] = string_field(team, "id") or None
            config["team_name"] = string_field(team, "name") or None
        enterprise = payload.get("enterprise")
        if isinstance(enterprise, dict):
            config["enterprise_id"] = string_field(enterprise, "id") or None

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
        load_oauth_provider(
            SlackOAuthProvider,
            client_id_env=SLACK_CLIENT_ID_ENV,
            client_secret_env=SLACK_CLIENT_SECRET_ENV,
            redirect_uri_env=SLACK_REDIRECT_URI_ENV,
        )
    ]


def _parse_scopes(value: object) -> list[str]:
    """Parse Slack's comma-delimited granted user scopes."""
    if not isinstance(value, str):
        return []
    return [scope.strip() for scope in value.split(",") if scope.strip()]
