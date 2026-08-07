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
"""GitHub OAuth provider."""

import os
from collections.abc import Mapping

import requests

from flwr.supercore.typing import JSONObject

from ..json_utils import required_string_field
from ..oauth import BaseOAuthProvider, load_oauth_provider

GITHUB_CONNECTOR_REF = "github"


class GitHubOAuthError(RuntimeError):
    """Secret-safe GitHub OAuth failure."""


class GitHubOAuthProvider(BaseOAuthProvider):
    """GitHub implementation of the OAuth provider contract."""

    connector_ref = GITHUB_CONNECTOR_REF
    display_name = "GitHub"
    description = "Search code and read files in public repositories."
    authorize_url = "https://github.com/login/oauth/authorize"
    error_type = GitHubOAuthError

    def authorization_parameters(
        self,
        *,
        redirect_uri: str,
        state: str,
        pkce_challenge: str | None,
    ) -> Mapping[str, str]:
        """Return public-information authorization parameters."""
        params = {
            "client_id": self._client_id,
            "redirect_uri": redirect_uri,
            "state": state,
        }
        if pkce_challenge is not None:
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
        """Exchange a GitHub authorization code for a token response."""
        data = {
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "code": code,
            "redirect_uri": redirect_uri,
        }
        if pkce_verifier is not None:
            data["code_verifier"] = pkce_verifier
        return requests.post(
            "https://github.com/login/oauth/access_token",
            headers={"Accept": "application/json"},
            data=data,
            timeout=30.0,
        )

    def parse_token_response(
        self, payload: JSONObject
    ) -> tuple[JSONObject, JSONObject]:
        """Extract a scope-free GitHub access token."""
        if "error" in payload:
            raise self._error("exchange failed")
        access_token = required_string_field(payload, "access_token", error=self._error)
        token_type = required_string_field(
            payload, "token_type", error=self._error
        ).lower()
        if token_type != "bearer" or payload.get("scope") not in (None, ""):
            raise self._error("returned unsupported permissions")
        return {"access_token": access_token}, {}


def get_configured_oauth_provider() -> GitHubOAuthProvider | None:
    """Return the configured GitHub OAuth provider, if available."""
    names = (
        "FLWR_GITHUB_CLIENT_ID",
        "FLWR_GITHUB_CLIENT_SECRET",
        "FLWR_GITHUB_REDIRECT_URI",
    )
    if not any(os.getenv(name, "").strip() for name in names):
        return None
    return load_oauth_provider(
        GitHubOAuthProvider,
        client_id_env=names[0],
        client_secret_env=names[1],
        redirect_uri_env=names[2],
    )
