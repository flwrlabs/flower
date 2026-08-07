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
from typing import TYPE_CHECKING, cast
from urllib.parse import urlencode

import requests

from flwr.supercore.typing import JSONObject

from .json_utils import required_string_field

if TYPE_CHECKING:
    from .definition import ProviderDefinition


class OAuthFlow:
    """Run OAuth from a provider's declarative configuration."""

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
        self._oauth = provider.oauth
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
        """Build an authorization URL from the provider definition."""
        params = {
            "client_id": self._client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
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
        return f"{self._oauth.authorization_url}?{urlencode(params)}"

    def exchange_code(
        self,
        *,
        code: str,
        redirect_uri: str,
        pkce_verifier: str | None,
    ) -> tuple[JSONObject, JSONObject]:
        """Exchange a code and extract standard credentials."""
        if not code:
            raise self._error("exchange failed")
        data = {
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        }
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
        try:
            response = requests.post(
                self._oauth.token_url,
                auth=auth,
                data=data,
                timeout=30.0,
            )
        except requests.RequestException:
            raise self._error("exchange failed") from None
        if response.status_code >= 400:
            raise self._error("exchange failed")
        try:
            response_payload = response.json()
        except ValueError:
            raise self._error("returned an invalid response") from None
        if not isinstance(response_payload, dict):
            raise self._error("returned an invalid response")
        return self._parse_token_response(cast(JSONObject, response_payload))

    def _parse_token_response(
        self, payload: JSONObject
    ) -> tuple[JSONObject, JSONObject]:
        """Validate and extract credentials from a token response."""
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
                item.strip()
                for item in scope.split(self._oauth.scope_separator)
                if item.strip()
            }
            if not set(self._oauth.scopes).issubset(granted):
                raise self._error("returned insufficient permissions")
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

    def _error(self, detail: str) -> RuntimeError:
        """Build a provider-labelled, secret-safe OAuth error."""
        return RuntimeError(f"{self.display_name} OAuth {detail}.")


def load_oauth_flow(provider: ProviderDefinition) -> OAuthFlow | None:
    """Return an OAuth flow loaded from its environment, if configured."""
    if provider.oauth is None:
        return None
    oauth = provider.oauth
    names = (oauth.client_id_env, oauth.client_secret_env, oauth.redirect_uri_env)
    if not any(os.getenv(name, "").strip() for name in names):
        return None
    return OAuthFlow(
        provider,
        client_id=os.getenv(names[0], ""),
        client_secret=os.getenv(names[1], ""),
        redirect_uri=os.getenv(names[2], ""),
    )
