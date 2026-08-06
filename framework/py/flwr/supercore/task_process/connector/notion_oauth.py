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
from collections.abc import Mapping

import requests

from flwr.supercore.typing import JSONObject

from .json_utils import required_string_field, string_field
from .notion import NOTION_API_VERSION, NOTION_CONNECTOR_REF
from .oauth import BaseOAuthProvider, load_oauth_provider

NOTION_CLIENT_ID_ENV = "FLWR_NOTION_CLIENT_ID"
NOTION_CLIENT_SECRET_ENV = "FLWR_NOTION_CLIENT_SECRET"
NOTION_REDIRECT_URI_ENV = "FLWR_NOTION_REDIRECT_URI"


class NotionOAuthError(RuntimeError):
    """Secret-safe Notion OAuth failure."""


class NotionOAuthProvider(BaseOAuthProvider):
    """Notion implementation of the OAuth provider contract."""

    connector_ref = NOTION_CONNECTOR_REF
    display_name = "Notion"
    description = "Search and read pages and data sources."
    authorize_url = "https://api.notion.com/v1/oauth/authorize"
    error_type = NotionOAuthError

    def authorization_parameters(
        self,
        *,
        redirect_uri: str,
        state: str,
        pkce_challenge: str | None,
    ) -> Mapping[str, str]:
        """Return Notion public-connection authorization parameters."""
        del pkce_challenge
        return {
            "client_id": self._client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "owner": "user",
            "state": state,
        }

    def request_token(
        self,
        *,
        code: str,
        redirect_uri: str,
        pkce_verifier: str | None,
    ) -> requests.Response:
        """Exchange a Notion authorization code for a token response."""
        del pkce_verifier
        return requests.post(
            "https://api.notion.com/v1/oauth/token",
            auth=(self._client_id, self._client_secret),
            headers={"Notion-Version": NOTION_API_VERSION},
            json={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
            },
            timeout=30.0,
        )

    def parse_token_response(
        self, payload: JSONObject
    ) -> tuple[JSONObject, JSONObject]:
        """Extract Notion credentials and workspace configuration."""
        if "error" in payload:
            raise self._error("exchange failed")
        credentials: JSONObject = {
            "access_token": required_string_field(
                payload, "access_token", error=self._error
            )
        }
        for key in ("refresh_token", "expires_in"):
            value = payload.get(key)
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                credentials[key] = value

        config: JSONObject = {}
        for key in ("workspace_id", "workspace_name", "bot_id"):
            if value := string_field(payload, key):
                config[key] = value
        owner = payload.get("owner")
        user = owner.get("user") if isinstance(owner, dict) else None
        if isinstance(user, dict):
            if owner_id := string_field(user, "id"):
                config["owner_user_id"] = owner_id
        return credentials, config


def get_configured_oauth_provider() -> NotionOAuthProvider | None:
    """Return the configured Notion OAuth provider, if available."""
    env_names = (
        NOTION_CLIENT_ID_ENV,
        NOTION_CLIENT_SECRET_ENV,
        NOTION_REDIRECT_URI_ENV,
    )
    if not any(os.getenv(name, "").strip() for name in env_names):
        return None
    return load_oauth_provider(
        NotionOAuthProvider,
        client_id_env=NOTION_CLIENT_ID_ENV,
        client_secret_env=NOTION_CLIENT_SECRET_ENV,
        redirect_uri_env=NOTION_REDIRECT_URI_ENV,
    )
