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
"""Provider-facing types and shared infrastructure for OAuth connector flows."""


import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Protocol, TypeVar, cast
from urllib.parse import urlencode

import requests

from flwr.supercore.typing import JSONObject


class OAuthConnectorProvider(Protocol):
    """Provider operations required by OAuth connector flows."""

    connector_ref: str
    display_name: str
    description: str

    def resolve_redirect_uri(self, requested_redirect_uri: str) -> str:
        """Validate and return the redirect URI to use for this OAuth flow."""

    def build_authorization_url(
        self,
        *,
        redirect_uri: str,
        state: str,
        pkce_challenge: str | None,
    ) -> str:
        """Return the provider authorization URL for a new OAuth session."""

    def exchange_code(
        self,
        *,
        code: str,
        redirect_uri: str,
        pkce_verifier: str | None,
    ) -> tuple[JSONObject, JSONObject]:
        """Exchange an authorization code for credentials and configuration."""


OAuthProviderT = TypeVar("OAuthProviderT", bound="BaseOAuthProvider")


def load_oauth_provider(
    provider_type: type[OAuthProviderT],
    *,
    client_id_env: str,
    client_secret_env: str,
    redirect_uri_env: str,
) -> OAuthProviderT:
    """Construct a provider from its environment configuration."""
    return provider_type(
        client_id=os.getenv(client_id_env, ""),
        client_secret=os.getenv(client_secret_env, ""),
        redirect_uri=os.getenv(redirect_uri_env, ""),
    )


class BaseOAuthProvider(ABC):
    """Implement the provider-independent OAuth authorization-code flow."""

    display_name: str
    authorize_url: str
    error_type: type[RuntimeError]

    def __init__(
        self, *, client_id: str, client_secret: str, redirect_uri: str
    ) -> None:
        values = (client_id.strip(), client_secret.strip(), redirect_uri.strip())
        if not all(values):
            raise ValueError(f"{self.display_name} OAuth configuration is incomplete.")
        self._client_id, self._client_secret, self._redirect_uri = values

    def resolve_redirect_uri(self, requested_redirect_uri: str) -> str:
        """Require the redirect URI configured for the provider application."""
        if requested_redirect_uri.strip() != self._redirect_uri:
            raise ValueError(f"{self.display_name} redirect URI is not allowed.")
        return self._redirect_uri

    def build_authorization_url(
        self,
        *,
        redirect_uri: str,
        state: str,
        pkce_challenge: str | None,
    ) -> str:
        """Build a provider authorization URL."""
        params = self.authorization_parameters(
            redirect_uri=redirect_uri,
            state=state,
            pkce_challenge=pkce_challenge,
        )
        return f"{self.authorize_url}?{urlencode(params)}"

    def exchange_code(
        self,
        *,
        code: str,
        redirect_uri: str,
        pkce_verifier: str | None,
    ) -> tuple[JSONObject, JSONObject]:
        """Exchange a code and parse its JSON object response."""
        if not code:
            raise self._error("exchange failed")
        try:
            response = self.request_token(
                code=code,
                redirect_uri=redirect_uri,
                pkce_verifier=pkce_verifier,
            )
        except requests.RequestException:
            raise self._error("exchange failed") from None
        if response.status_code >= 400:
            raise self._error("exchange failed")
        try:
            payload = response.json()
        except ValueError:
            raise self._error("returned an invalid response") from None
        if not isinstance(payload, dict):
            raise self._error("returned an invalid response")
        return self.parse_token_response(cast(JSONObject, payload))

    def _error(self, detail: str) -> RuntimeError:
        """Build a provider-specific secret-safe error."""
        return self.error_type(f"{self.display_name} OAuth {detail}.")

    @abstractmethod
    def authorization_parameters(
        self,
        *,
        redirect_uri: str,
        state: str,
        pkce_challenge: str | None,
    ) -> Mapping[str, str]:
        """Return provider-specific authorization parameters."""

    @abstractmethod
    def request_token(
        self,
        *,
        code: str,
        redirect_uri: str,
        pkce_verifier: str | None,
    ) -> requests.Response:
        """Send the provider-specific token request."""

    @abstractmethod
    def parse_token_response(
        self, payload: JSONObject
    ) -> tuple[JSONObject, JSONObject]:
        """Parse provider-specific credentials and configuration."""
