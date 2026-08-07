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

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, TypeVar, cast
from urllib.parse import urlencode

import requests

from flwr.supercore.typing import JSONObject

from .json_utils import required_string_field

if TYPE_CHECKING:
    from .definition import ProviderDefinition


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


class DeclarativeOAuthProvider(BaseOAuthProvider):
    """Implement OAuth from a provider's declarative configuration."""

    error_type = RuntimeError

    def __init__(
        self,
        provider: ProviderDefinition,
        *,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
    ) -> None:
        if provider.oauth is None:
            raise ValueError(f"Provider '{provider.ref}' does not define OAuth.")
        self.connector_ref = provider.ref
        self.display_name = provider.display_name
        self.description = provider.description
        self.authorize_url = provider.oauth.authorization_url
        self._oauth = provider.oauth
        super().__init__(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
        )

    def authorization_parameters(
        self,
        *,
        redirect_uri: str,
        state: str,
        pkce_challenge: str | None,
    ) -> Mapping[str, str]:
        """Return authorization parameters from the provider definition."""
        params = {
            "client_id": self._client_id,
            "redirect_uri": redirect_uri,
            "state": state,
        }
        if self._oauth.scopes:
            params[self._oauth.scope_parameter] = self._oauth.scope_separator.join(
                self._oauth.scopes
            )
        if self._oauth.use_pkce:
            if not pkce_challenge:
                raise ValueError(f"{self.display_name} OAuth requires PKCE.")
            params.update(
                code_challenge=pkce_challenge,
                code_challenge_method="S256",
            )
        return params

    def request_token(
        self,
        *,
        code: str,
        redirect_uri: str,
        pkce_verifier: str | None,
    ) -> requests.Response:
        """Exchange an authorization code using the provider definition."""
        data = {"code": code, "redirect_uri": redirect_uri}
        if self._oauth.use_pkce:
            if not pkce_verifier:
                raise self._error("exchange failed")
            data["code_verifier"] = pkce_verifier
        auth = None
        if self._oauth.token_auth_method == "client_secret_basic":
            auth = (self._client_id, self._client_secret)
        else:
            data.update(
                client_id=self._client_id,
                client_secret=self._client_secret,
            )
        return requests.post(
            self._oauth.token_url,
            auth=auth,
            data=data,
            timeout=30.0,
        )

    def parse_token_response(
        self, payload: JSONObject
    ) -> tuple[JSONObject, JSONObject]:
        """Extract standard credentials from the configured response path."""
        if (
            "error" in payload
            or self._oauth.success_field
            and payload.get(self._oauth.success_field) is not True
        ):
            raise self._error("exchange failed")
        token_payload = payload
        for key in self._oauth.token_response_path:
            value = token_payload.get(key)
            if not isinstance(value, dict):
                raise self._error("returned an invalid response")
            token_payload = value
        if "error" in token_payload:
            raise self._error("exchange failed")
        scope = token_payload.get("scope")
        if scope is not None:
            if not isinstance(scope, str):
                raise self._error("returned an invalid response")
            granted = {
                item for item in scope.split(self._oauth.scope_separator) if item
            }
            if not granted.issubset(self._oauth.scopes):
                raise self._error("returned unsupported permissions")
        credentials: JSONObject = {
            "access_token": required_string_field(
                token_payload, "access_token", error=self._error
            )
        }
        for key in ("refresh_token", "expires_in", "token_type"):
            value = token_payload.get(key)
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                credentials[key] = value
        return credentials, {}


def load_declarative_oauth_provider(
    provider: ProviderDefinition,
) -> DeclarativeOAuthProvider | None:
    """Return a provider loaded from its environment, if configured."""
    if provider.oauth is None:
        return None
    oauth = provider.oauth
    names = (oauth.client_id_env, oauth.client_secret_env, oauth.redirect_uri_env)
    if not any(os.getenv(name, "").strip() for name in names):
        return None
    return DeclarativeOAuthProvider(
        provider,
        client_id=os.getenv(names[0], ""),
        client_secret=os.getenv(names[1], ""),
        redirect_uri=os.getenv(names[2], ""),
    )
