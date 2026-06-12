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
"""Abstract classes for Flower account auth plugins."""


from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from flwr.supercore.auth.typing import (
    AccountAuthCredentials,
    AccountAuthLoginDetails,
    AccountInfo,
)


class ControlAuthnPlugin(ABC):
    """Abstract Flower Authentication Plugin class for ControlServicer.

    Parameters
    ----------
    account_auth_config_path : Path
        Path to the YAML file containing the authentication configuration.
    verify_tls_cert : bool
        Boolean indicating whether to verify the TLS certificate
        when making requests to the server.
    """

    @abstractmethod
    def __init__(
        self,
        account_auth_config_path: Path,
        verify_tls_cert: bool,
    ):
        """Abstract constructor."""

    @abstractmethod
    def get_login_details(self) -> AccountAuthLoginDetails | None:
        """Get the login details."""

    @abstractmethod
    def validate_tokens_in_metadata(
        self, metadata: Sequence[tuple[str, str | bytes]]
    ) -> tuple[bool, AccountInfo | None]:
        """Validate authentication tokens in the provided metadata."""

    @abstractmethod
    def get_auth_tokens(self, device_code: str) -> AccountAuthCredentials | None:
        """Get authentication tokens."""

    @abstractmethod
    def refresh_tokens(
        self, metadata: Sequence[tuple[str, str | bytes]]
    ) -> tuple[Sequence[tuple[str, str | bytes]] | None, AccountInfo | None]:
        """Refresh authentication tokens in the provided metadata."""


class ControlAuthzPlugin(ABC):  # pylint: disable=too-few-public-methods
    """Abstract Flower Authorization Plugin class for ControlServicer.

    Parameters
    ----------
    account_auth_config_path : Path
        Path to the YAML file containing the authorization configuration.
    verify_tls_cert : bool
        Boolean indicating whether to verify the TLS certificate
        when making requests to the server.
    """

    @abstractmethod
    def __init__(self, account_auth_config_path: Path, verify_tls_cert: bool):
        """Abstract constructor."""

    @abstractmethod
    def authorize(self, account_info: AccountInfo) -> bool:
        """Verify account authorization request."""
