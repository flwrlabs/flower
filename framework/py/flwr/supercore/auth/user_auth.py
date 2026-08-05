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
"""Shared HTTP user-authentication contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from flwr.supercore.error import ApiErrorCode, FlowerError

from .typing import AccountAuthCredentials, AccountAuthLoginDetails, AccountInfo


@runtime_checkable
class UserAuthenticationService(Protocol):
    """Authenticate HTTP users and manage their OIDC token flow."""

    async def authenticate_user(self, access_token: str) -> AccountInfo:
        """Authenticate and resolve a provisioned Flower account."""

    async def authenticate_for_bootstrap(self, access_token: str) -> str:
        """Authenticate a user without requiring a provisioned account."""

    async def refresh_tokens(self, refresh_token: str) -> AccountAuthCredentials:
        """Exchange a refresh token for a new token pair."""

    async def start_device_authorization(self) -> AccountAuthLoginDetails:
        """Start an OIDC device-authorization flow."""

    async def exchange_device_code(self, device_code: str) -> AccountAuthCredentials:
        """Exchange an OIDC device code for a token pair."""


def get_bearer_token(authorization: str | None) -> str:
    """Return the token from a strict HTTP Bearer authorization header."""
    if authorization is None:
        raise FlowerError(
            ApiErrorCode.ACCOUNT_AUTHENTICATION_FAILED,
            "Missing Authorization header.",
        )

    scheme, separator, token = authorization.partition(" ")
    if (
        not separator
        or scheme.lower() != "bearer"
        or not token
        or token != token.strip()
        or any(character.isspace() for character in token)
    ):
        raise FlowerError(
            ApiErrorCode.ACCOUNT_AUTHENTICATION_FAILED,
            "Authorization header must contain exactly one Bearer token.",
        )
    return token
