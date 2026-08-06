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
from collections.abc import Callable, Mapping
from typing import Protocol, TypeVar, cast
from urllib.parse import urlencode

import requests

from flwr.supercore.typing import JSONObject

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


class DeclarativeOAuthProvider:
    """Implement a standard OAuth flow from a provider definition."""

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
        values = (client_id.strip(), client_secret.strip(), redirect_uri.strip())
        if (
            not values[0]
            or not values[2]
            or (provider.oauth.token_auth_method != "none" and not values[1])
        ):
            raise ValueError(
                f"{provider.display_name} OAuth configuration is incomplete."
            )
        self.connector_ref = provider.ref
        self.display_name = provider.display_name
        self.description = provider.description
        self._oauth = provider.oauth
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
        """Build an authorization URL from declarative OAuth settings."""
        params = _string_mapping(self._oauth.authorization_params)
        params.update(
            {
                "client_id": self._client_id,
                "redirect_uri": redirect_uri,
                "response_type": "code",
                "state": state,
            }
        )
        if self._oauth.scopes:
            params["scope"] = self._oauth.scope_separator.join(self._oauth.scopes)
        if self._oauth.use_pkce:
            if not pkce_challenge:
                raise ValueError(f"{self.display_name} OAuth requires PKCE.")
            params.update(
                {"code_challenge": pkce_challenge, "code_challenge_method": "S256"}
            )
        return f"{self._oauth.authorization_url}?{urlencode(params)}"

    def exchange_code(
        self,
        *,
        code: str,
        redirect_uri: str,
        pkce_verifier: str | None,
    ) -> tuple[JSONObject, JSONObject]:
        """Exchange an authorization code and extract standard token fields."""
        if not code:
            raise self._error("exchange failed")
        body: JSONObject = {
            **self._oauth.token_params,
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
        }
        if self._oauth.use_pkce:
            if not pkce_verifier:
                raise self._error("exchange failed")
            body["code_verifier"] = pkce_verifier
        auth: tuple[str, str] | None = None
        if self._oauth.token_auth_method == "client_secret_basic":
            auth = (self._client_id, self._client_secret)
        elif self._oauth.token_auth_method == "client_secret_post":
            body.update(
                {"client_id": self._client_id, "client_secret": self._client_secret}
            )
        else:
            body["client_id"] = self._client_id
        response = self._request_token(body, auth)
        token_payload = self._read_token_payload(response)
        return self._extract_token_fields(token_payload)

    def _request_token(
        self, body: JSONObject, auth: tuple[str, str] | None
    ) -> requests.Response:
        """Send one token request using the configured encoding."""
        try:
            if self._oauth.token_request_format == "json":
                return requests.post(
                    self._oauth.token_url, auth=auth, json=body, timeout=30.0
                )
            return requests.post(
                self._oauth.token_url, auth=auth, data=body, timeout=30.0
            )
        except requests.RequestException:
            raise self._error("exchange failed") from None

    def _read_token_payload(self, response: requests.Response) -> JSONObject:
        """Validate and unwrap one token response."""
        if response.status_code >= 400:
            raise self._error("exchange failed")
        try:
            payload = response.json()
        except ValueError:
            raise self._error("returned an invalid response") from None
        if not isinstance(payload, dict):
            raise self._error("returned an invalid response")
        token_payload = _nested_object(
            cast(JSONObject, payload), self._oauth.token_response_path, self._error
        )
        if "error" in token_payload:
            raise self._error("exchange failed")
        return token_payload

    def _extract_token_fields(
        self, token_payload: JSONObject
    ) -> tuple[JSONObject, JSONObject]:
        """Separate credentials from non-secret provider configuration."""
        access_token = token_payload.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise self._error("returned an invalid response")
        credentials: JSONObject = {"access_token": access_token}
        for key in ("refresh_token", "expires_in", "token_type"):
            value = token_payload.get(key)
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                credentials[key] = value
        config: JSONObject = {}
        for key in self._oauth.config_fields:
            value = token_payload.get(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                config[key] = value
        return credentials, config

    def _error(self, detail: str) -> RuntimeError:
        """Build a provider-labelled, secret-safe OAuth error."""
        return RuntimeError(f"{self.display_name} OAuth {detail}.")


def load_declarative_oauth_provider(
    provider: ProviderDefinition,
) -> DeclarativeOAuthProvider | None:
    """Load a provider's OAuth app from its declared environment variables."""
    if provider.oauth is None:
        return None
    oauth = provider.oauth
    values = (
        os.getenv(oauth.client_id_env, ""),
        os.getenv(oauth.client_secret_env, "") if oauth.client_secret_env else "",
        os.getenv(oauth.redirect_uri_env, ""),
    )
    if not any(value.strip() for value in values):
        return None
    return DeclarativeOAuthProvider(
        provider,
        client_id=values[0],
        client_secret=values[1],
        redirect_uri=values[2],
    )


def _string_mapping(value: Mapping[str, object]) -> dict[str, str]:
    """Validate provider-defined static request parameters."""
    if not all(isinstance(item, str) for item in value.values()):
        raise ValueError("OAuth static parameters must be strings.")
    return {key: cast(str, item) for key, item in value.items()}


def _nested_object(
    payload: JSONObject,
    path: tuple[str, ...],
    error: Callable[[str], RuntimeError],
) -> JSONObject:
    """Read a configured object envelope from an OAuth response."""
    current = payload
    for key in path:
        value = current.get(key)
        if not isinstance(value, dict):
            raise error("returned an invalid response")
        current = value
    return current
