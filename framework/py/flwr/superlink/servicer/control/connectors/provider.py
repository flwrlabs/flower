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
"""Provider-facing types for Control OAuth connector flows."""


from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from flwr.supercore.typing import JSONObject


@dataclass(frozen=True)
class ConnectorDefinition:
    """User-connectable OAuth connector metadata."""

    connector_ref: str
    display_name: str
    description: str
    supports_pkce: bool = False


@dataclass(frozen=True)
class ConnectorOAuthResult:
    """Credentials and configuration produced by an OAuth code exchange."""

    credentials: JSONObject
    config: JSONObject


class ConnectorOAuthProvider(Protocol):
    """Provider operations required by the Control OAuth handlers."""

    definition: ConnectorDefinition

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
    ) -> ConnectorOAuthResult:
        """Exchange an authorization code for credentials and configuration."""


def normalize_connector_ref(connector_ref: str) -> str:
    """Return the canonical representation of a connector reference."""
    return connector_ref.strip().lower()


def make_connector_oauth_provider_map(
    providers: Sequence[ConnectorOAuthProvider],
) -> dict[str, ConnectorOAuthProvider]:
    """Index OAuth providers by canonical connector reference."""
    provider_map: dict[str, ConnectorOAuthProvider] = {}
    for provider in providers:
        connector_ref = normalize_connector_ref(provider.definition.connector_ref)
        if not connector_ref:
            raise ValueError("Connector provider has an empty connector reference.")
        if connector_ref in provider_map:
            raise ValueError(
                f"Duplicate connector OAuth provider for reference '{connector_ref}'."
            )
        provider_map[connector_ref] = provider
    return provider_map
