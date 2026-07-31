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
"""Notion OAuth provider."""

import os
from urllib.parse import urlencode

import requests

from flwr.supercore.typing import JSONObject

from .notion import NOTION_API_VERSION, NOTION_CONNECTOR_REF
from .oauth import OAuthConnectorProvider

NOTION_CLIENT_ID_ENV = "FLWR_NOTION_CLIENT_ID"
NOTION_CLIENT_SECRET_ENV = "FLWR_NOTION_CLIENT_SECRET"
NOTION_REDIRECT_URI_ENV = "FLWR_NOTION_REDIRECT_URI"

_NOTION_AUTHORIZE_URL = "https://api.notion.com/v1/oauth/authorize"
_NOTION_TOKEN_URL = "https://api.notion.com/v1/oauth/token"
_REQUEST_TIMEOUT = 30.0


class NotionOAuthError(RuntimeError):
    """Secret-safe Notion OAuth failure."""


class NotionOAuthProvider:
    """Notion implementation of the OAuth provider contract."""

    connector_ref = NOTION_CONNECTOR_REF
    display_name = "Notion"
    description = "Search and read pages and data sources."

    def __init__(
        self, *, client_id: str, client_secret: str, redirect_uri: str
    ) -> None:
        client_id = client_id.strip()
        client_secret = client_secret.strip()
        redirect_uri = redirect_uri.strip()
        if not client_id or not client_secret or not redirect_uri:
            raise ValueError("Notion OAuth configuration is incomplete.")
        self._client_id = client_id
        self._client_secret = client_secret
        self._redirect_uri = redirect_uri

    def resolve_redirect_uri(self, requested_redirect_uri: str) -> str:
        """Require the redirect URI configured for the Notion connection."""
        if requested_redirect_uri.strip() != self._redirect_uri:
            raise ValueError("Notion redirect URI is not allowed.")
        return self._redirect_uri

    def build_authorization_url(
        self,
        *,
        redirect_uri: str,
        state: str,
        pkce_challenge: str | None,
    ) -> str:
        """Build a Notion public-connection authorization URL."""
        # Notion's public REST OAuth flow does not document PKCE parameters.
        del pkce_challenge
        params = {
            "client_id": self._client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "owner": "user",
            "state": state,
        }
        return f"{_NOTION_AUTHORIZE_URL}?{urlencode(params)}"

    def exchange_code(
        self,
        *,
        code: str,
        redirect_uri: str,
        pkce_verifier: str | None,
    ) -> tuple[JSONObject, JSONObject]:
        """Exchange a Notion authorization code for connection credentials."""
        # Notion's public REST OAuth flow does not document PKCE parameters.
        del pkce_verifier
        if not code:
            raise NotionOAuthError("Notion OAuth exchange failed.")
        try:
            response = requests.post(
                _NOTION_TOKEN_URL,
                auth=(self._client_id, self._client_secret),
                headers={
                    "Accept": "application/json",
                    "Notion-Version": NOTION_API_VERSION,
                },
                json={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
                timeout=_REQUEST_TIMEOUT,
            )
        except requests.RequestException:
            raise NotionOAuthError("Notion OAuth exchange failed.") from None
        if response.status_code >= 400:
            raise NotionOAuthError("Notion OAuth exchange failed.")

        try:
            payload = response.json()
        except ValueError:
            raise NotionOAuthError(
                "Notion OAuth returned an invalid response."
            ) from None
        if not isinstance(payload, dict):
            raise NotionOAuthError("Notion OAuth returned an invalid response.")
        if "error" in payload:
            raise NotionOAuthError("Notion OAuth exchange failed.")

        access_token = _required_token(payload, "access_token")
        credentials: JSONObject = {"access_token": access_token}
        refresh_token = _optional_string(payload.get("refresh_token"))
        if refresh_token is not None:
            credentials["refresh_token"] = refresh_token
        expires_in = payload.get("expires_in")
        if isinstance(expires_in, int) and not isinstance(expires_in, bool):
            credentials["expires_in"] = expires_in

        config: JSONObject = {}
        for key in ("workspace_id", "workspace_name", "bot_id"):
            value = _optional_string(payload.get(key))
            if value is not None:
                config[key] = value
        owner_user_id = _owner_user_id(payload.get("owner"))
        if owner_user_id is not None:
            config["owner_user_id"] = owner_user_id
        return credentials, config


def get_configured_connector_oauth_providers() -> list[OAuthConnectorProvider]:
    """Return the configured Notion OAuth provider, if available."""
    values = {
        NOTION_CLIENT_ID_ENV: os.getenv(NOTION_CLIENT_ID_ENV, "").strip(),
        NOTION_CLIENT_SECRET_ENV: os.getenv(NOTION_CLIENT_SECRET_ENV, "").strip(),
        NOTION_REDIRECT_URI_ENV: os.getenv(NOTION_REDIRECT_URI_ENV, "").strip(),
    }
    if not any(values.values()):
        return []
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError(
            "Notion OAuth configuration requires environment variables: "
            + ", ".join(missing)
            + "."
        )
    return [
        NotionOAuthProvider(
            client_id=values[NOTION_CLIENT_ID_ENV],
            client_secret=values[NOTION_CLIENT_SECRET_ENV],
            redirect_uri=values[NOTION_REDIRECT_URI_ENV],
        )
    ]


def _required_token(payload: dict[object, object], key: str) -> str:
    """Read a required token without including its value in errors."""
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise NotionOAuthError("Notion OAuth response is missing credentials.")
    return value


def _optional_string(value: object) -> str | None:
    """Return a non-empty string or None."""
    return value if isinstance(value, str) and value else None


def _owner_user_id(value: object) -> str | None:
    """Extract the stable authorizing user ID from Notion owner metadata."""
    if not isinstance(value, dict):
        return None
    user = value.get("user")
    if not isinstance(user, dict):
        return None
    return _optional_string(user.get("id"))
