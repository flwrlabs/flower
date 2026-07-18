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
"""Provider-facing types for OAuth connector flows."""


from typing import Protocol

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
