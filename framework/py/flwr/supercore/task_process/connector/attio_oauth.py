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
"""Attio OAuth provider."""

import os
from urllib.parse import urlencode

import requests

from flwr.supercore.typing import JSONObject

from .attio import ATTIO_CONNECTOR_REF
from .oauth import OAuthConnectorProvider

ATTIO_CLIENT_ID_ENV = "FLWR_ATTIO_CLIENT_ID"
ATTIO_CLIENT_SECRET_ENV = "FLWR_ATTIO_CLIENT_SECRET"
ATTIO_REDIRECT_URI_ENV = "FLWR_ATTIO_REDIRECT_URI"

_ATTIO_AUTHORIZE_URL = "https://app.attio.com/authorize"
_ATTIO_TOKEN_URL = "https://app.attio.com/oauth/token"
_REQUEST_TIMEOUT = 30.0


class AttioOAuthError(RuntimeError):
    """Secret-safe Attio OAuth failure."""


class AttioOAuthProvider:
    """Attio implementation of the OAuth provider contract."""

    connector_ref = ATTIO_CONNECTOR_REF
    display_name = "Attio"
    description = "Search records and read meeting transcripts."

    def __init__(
        self, *, client_id: str, client_secret: str, redirect_uri: str
    ) -> None:
        client_id = client_id.strip()
        client_secret = client_secret.strip()
        redirect_uri = redirect_uri.strip()
        if not client_id or not client_secret or not redirect_uri:
            raise ValueError("Attio OAuth configuration is incomplete.")
        self._client_id = client_id
        self._client_secret = client_secret
        self._redirect_uri = redirect_uri

    def resolve_redirect_uri(self, requested_redirect_uri: str) -> str:
        """Require the redirect URI configured for the Attio app."""
        if requested_redirect_uri.strip() != self._redirect_uri:
            raise ValueError("Attio redirect URI is not allowed.")
        return self._redirect_uri

    def build_authorization_url(
        self,
        *,
        redirect_uri: str,
        state: str,
        pkce_challenge: str | None,
    ) -> str:
        """Build an Attio authorization-code URL."""
        # Attio's OAuth reference does not document PKCE parameters.
        del pkce_challenge
        params = {
            "response_type": "code",
            "client_id": self._client_id,
            "redirect_uri": self.resolve_redirect_uri(redirect_uri),
            "state": state,
        }
        return f"{_ATTIO_AUTHORIZE_URL}?{urlencode(params)}"

    def exchange_code(
        self,
        *,
        code: str,
        redirect_uri: str,
        pkce_verifier: str | None,
    ) -> tuple[JSONObject, JSONObject]:
        """Exchange an authorization code for Attio credentials."""
        del pkce_verifier
        try:
            response = requests.post(
                _ATTIO_TOKEN_URL,
                data={
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": self.resolve_redirect_uri(redirect_uri),
                },
                timeout=_REQUEST_TIMEOUT,
            )
        except requests.RequestException:
            raise AttioOAuthError("Attio OAuth exchange failed.") from None
        if response.status_code >= 400:
            raise AttioOAuthError("Attio OAuth exchange failed.")
        try:
            payload = response.json()
        except ValueError:
            raise AttioOAuthError("Attio OAuth response is invalid.") from None
        if not isinstance(payload, dict):
            raise AttioOAuthError("Attio OAuth response is invalid.")
        access_token = payload.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise AttioOAuthError("Attio OAuth response is missing credentials.")
        credentials: JSONObject = {"access_token": access_token}
        refresh_token = payload.get("refresh_token")
        if isinstance(refresh_token, str) and refresh_token:
            credentials["refresh_token"] = refresh_token
        return credentials, {}


def get_configured_connector_oauth_providers() -> list[OAuthConnectorProvider]:
    """Return the configured Attio OAuth provider, if available."""
    values = {
        ATTIO_CLIENT_ID_ENV: os.getenv(ATTIO_CLIENT_ID_ENV, "").strip(),
        ATTIO_CLIENT_SECRET_ENV: os.getenv(ATTIO_CLIENT_SECRET_ENV, "").strip(),
        ATTIO_REDIRECT_URI_ENV: os.getenv(ATTIO_REDIRECT_URI_ENV, "").strip(),
    }
    if not any(values.values()):
        return []
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError(
            "Attio OAuth configuration requires environment variables: "
            + ", ".join(missing)
            + "."
        )
    return [
        AttioOAuthProvider(
            client_id=values[ATTIO_CLIENT_ID_ENV],
            client_secret=values[ATTIO_CLIENT_SECRET_ENV],
            redirect_uri=values[ATTIO_REDIRECT_URI_ENV],
        )
    ]
