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
# ===============================================================================
"""Attio OAuth provider."""

import os
from collections.abc import Mapping

import requests

from flwr.supercore.typing import JSONObject

from ..json_utils import required_string_field
from ..oauth import BaseOAuthProvider, load_oauth_provider

ATTIO_CONNECTOR_REF = "attio"

_ATTIO_AUTHORIZE_URL = "https://app.attio.com/authorize"
_ATTIO_TOKEN_URL = "https://app.attio.com/oauth/token"


class AttioOAuthError(RuntimeError):
    """Secret-safe Attio OAuth failure."""


class AttioOAuthProvider(BaseOAuthProvider):
    """Attio implementation of the OAuth provider contract."""

    connector_ref = ATTIO_CONNECTOR_REF
    display_name = "Attio"
    description = "Search records and read meeting transcripts."
    authorize_url = _ATTIO_AUTHORIZE_URL
    error_type = AttioOAuthError

    def authorization_parameters(
        self,
        *,
        redirect_uri: str,
        state: str,
        pkce_challenge: str | None,
    ) -> Mapping[str, str]:
        """Return Attio authorization parameters."""
        if pkce_challenge is not None:
            raise ValueError("Attio PKCE is not enabled for this provider.")
        return {
            "response_type": "code",
            "client_id": self._client_id,
            "redirect_uri": self.resolve_redirect_uri(redirect_uri),
            "state": state,
        }

    def request_token(
        self,
        *,
        code: str,
        redirect_uri: str,
        pkce_verifier: str | None,
    ) -> requests.Response:
        """Exchange an Attio authorization code for a token response."""
        del pkce_verifier
        return requests.post(
            _ATTIO_TOKEN_URL,
            data={
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.resolve_redirect_uri(redirect_uri),
            },
            timeout=30.0,
        )

    def parse_token_response(
        self, payload: JSONObject
    ) -> tuple[JSONObject, JSONObject]:
        """Extract Attio credentials from a token response."""
        credentials: JSONObject = {
            "access_token": required_string_field(
                payload, "access_token", error=self._error
            )
        }
        refresh_token = payload.get("refresh_token")
        if isinstance(refresh_token, str) and refresh_token:
            credentials["refresh_token"] = refresh_token
        return credentials, {}


def get_configured_oauth_provider() -> AttioOAuthProvider | None:
    """Return the configured Attio OAuth provider, if available."""
    names = (
        "FLWR_ATTIO_CLIENT_ID",
        "FLWR_ATTIO_CLIENT_SECRET",
        "FLWR_ATTIO_REDIRECT_URI",
    )
    if not any(os.getenv(name, "").strip() for name in names):
        return None
    return load_oauth_provider(
        AttioOAuthProvider,
        client_id_env=names[0],
        client_secret_env=names[1],
        redirect_uri_env=names[2],
    )
