# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Concrete NoOp implementation for Servicer-side account authentication."""


from collections.abc import Sequence

from flwr.common.constant import NOOP_ACCOUNT_NAME, NOOP_FLWR_AID, AuthnType
from flwr.supercore.auth.typing import (
    AccountAuthCredentials,
    AccountAuthLoginDetails,
    AccountInfo,
)
from flwr.supercore.error import ApiErrorCode, FlowerError

from .auth_plugin import ControlAuthnPlugin


def _create_noop_account_info() -> AccountInfo:
    """Create account information without sharing mutable state between calls."""
    return AccountInfo(
        flwr_aid=NOOP_FLWR_AID,
        account_name=NOOP_ACCOUNT_NAME,
    )


class NoOpControlAuthnPlugin(ControlAuthnPlugin):
    """No-operation implementation of ControlAuthnPlugin."""

    def __init__(self) -> None:
        pass

    def get_login_details(self) -> AccountAuthLoginDetails | None:
        """Get the login details."""
        # This allows the `flwr login` command to load the NoOp plugin accordingly,
        # which then raises a LoginError when attempting to login.
        return AccountAuthLoginDetails(
            authn_type=AuthnType.NOOP,  # No operation authn type
            device_code="",
            verification_uri_complete="",
            expires_in=0,
            interval=0,
        )

    def validate_tokens_in_metadata(
        self, metadata: Sequence[tuple[str, str | bytes]]
    ) -> tuple[bool, AccountInfo | None]:
        """Return valid for no-op plugin."""
        return True, _create_noop_account_info()

    def get_auth_tokens(self, device_code: str) -> AccountAuthCredentials | None:
        """Get authentication tokens."""
        raise FlowerError(
            ApiErrorCode.NO_ACCOUNT_AUTH,
            "Account authentication is not enabled on this SuperLink.",
        )

    def refresh_tokens(
        self, metadata: Sequence[tuple[str, str | bytes]]
    ) -> tuple[Sequence[tuple[str, str | bytes]] | None, AccountInfo | None]:
        """Refresh authentication tokens in the provided metadata."""
        return metadata, _create_noop_account_info()
