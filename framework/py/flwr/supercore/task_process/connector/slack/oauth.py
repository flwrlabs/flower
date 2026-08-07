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

from ..json_utils import object_field, required_string_field
from ..oauth import BaseOAuthProvider, load_oauth_provider

SLACK_CONNECTOR_REF = "slack"
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
    authorize_url = "https://slack.com/oauth/v2/authorize"
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
            "https://slack.com/api/oauth.v2.access",
            auth=(self._client_id, self._client_secret),
            data=data,
            timeout=30.0,
        )

    def parse_token_response(
        self, payload: JSONObject
    ) -> tuple[JSONObject, JSONObject]:
        """Extract Slack user credentials."""
        if payload.get("ok") is not True:
            raise SlackOAuthError("Slack OAuth exchange failed.")
        authed_user = object_field(payload, "authed_user", error=self._error)
        credentials: JSONObject = {
            "access_token": required_string_field(
                authed_user, "access_token", error=self._error
            )
        }
        refresh_token = authed_user.get("refresh_token")
        if isinstance(refresh_token, str) and refresh_token:
            credentials["refresh_token"] = refresh_token
        expires_in = authed_user.get("expires_in")
        if isinstance(expires_in, int) and not isinstance(expires_in, bool):
            credentials["expires_in"] = expires_in
        return credentials, {}


def get_configured_oauth_provider() -> SlackOAuthProvider | None:
    """Return the configured Slack OAuth provider, if available."""
    names = (
        "FLWR_SLACK_CLIENT_ID",
        "FLWR_SLACK_CLIENT_SECRET",
        "FLWR_SLACK_REDIRECT_URI",
    )
    if not any(os.getenv(name, "").strip() for name in names):
        return None
    return load_oauth_provider(
        SlackOAuthProvider,
        client_id_env=names[0],
        client_secret_env=names[1],
        redirect_uri_env=names[2],
    )
