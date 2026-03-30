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
"""Flower CLI account auth plugin for OIDC."""


import time
import webbrowser
from collections.abc import Sequence

import click
import typer

from flwr.cli.constant import (
    ACCESS_TOKEN_STORE_KEY,
    AUTHN_TYPE_STORE_KEY,
    REFRESH_TOKEN_STORE_KEY,
)
from flwr.common.constant import ACCESS_TOKEN_KEY, REFRESH_TOKEN_KEY, AuthnType
from flwr.common.typing import AccountAuthCredentials, AccountAuthLoginDetails
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    GetAuthTokensRequest,
    GetAuthTokensResponse,
)
from flwr.proto.control_pb2_grpc import ControlStub
from flwr.supercore.credential_store import get_credential_store

from .auth_plugin import CliAuthPlugin, LoginError


class OidcCliPlugin(CliAuthPlugin):
    """Flower OIDC authentication plugin for CLI.

    This plugin implements OpenID Connect (OIDC) device flow authentication for CLI
    access to Flower SuperLink.
    """

    def __init__(self, host: str):
        self.access_token: str | None = None
        self.refresh_token: str | None = None
        self.host = host
        self.store = get_credential_store()

    @staticmethod
    def login(
        login_details: AccountAuthLoginDetails,
        control_stub: ControlStub,
    ) -> AccountAuthCredentials:
        """Authenticate the account and retrieve authentication credentials.

        Parameters
        ----------
        login_details : AccountAuthLoginDetails
            Login details containing device code and verification URI.
        control_stub : ControlStub
            Control stub for making authentication requests.

        Returns
        -------
        AccountAuthCredentials
            The access and refresh tokens.

        Raises
        ------
        LoginError
            If authentication times out.
        """
        # Prompt user to login via browser
        webbrowser.open(login_details.verification_uri_complete)
        typer.secho(
            "A browser window has been opened for you to "
            "log into your Flower account.\n"
            "If it did not open automatically, use this URL:\n"
            f"{login_details.verification_uri_complete}",
            fg=typer.colors.BLUE,
        )

        # Wait for user to complete login
        start_time = time.time()
        time.sleep(login_details.interval)

        while (time.time() - start_time) < login_details.expires_in:
            res: GetAuthTokensResponse = control_stub.GetAuthTokens(
                GetAuthTokensRequest(device_code=login_details.device_code)
            )

            access_token = res.access_token
            refresh_token = res.refresh_token

            if access_token and refresh_token:
                return AccountAuthCredentials(
                    access_token=access_token,
                    refresh_token=refresh_token,
                )

            time.sleep(login_details.interval)

        raise LoginError("Process timed out.")

    def store_tokens(self, credentials: AccountAuthCredentials) -> None:
        """Store authentication tokens to the credential store."""
        host = self.host
        # Retrieve tokens
        access_token = credentials.access_token
        refresh_token = credentials.refresh_token

        # Store tokens in the credential store
        self.store.set(AUTHN_TYPE_STORE_KEY % host, AuthnType.OIDC.encode("utf-8"))
        self.store.set(ACCESS_TOKEN_STORE_KEY % host, access_token.encode("utf-8"))
        self.store.set(REFRESH_TOKEN_STORE_KEY % host, refresh_token.encode("utf-8"))

        # Update internal state
        self.access_token = access_token
        self.refresh_token = refresh_token

    def load_tokens(self) -> None:
        """Load authentication tokens from the credential store."""
        access_token_bytes = self.store.get(ACCESS_TOKEN_STORE_KEY % self.host)
        refresh_token_bytes = self.store.get(REFRESH_TOKEN_STORE_KEY % self.host)

        if access_token_bytes is not None and refresh_token_bytes is not None:
            self.access_token = access_token_bytes.decode("utf-8")
            self.refresh_token = refresh_token_bytes.decode("utf-8")

    def write_tokens_to_metadata(
        self, metadata: Sequence[tuple[str, str | bytes]]
    ) -> Sequence[tuple[str, str | bytes]]:
        """Write authentication tokens to the provided metadata."""
        if self.access_token is None or self.refresh_token is None:
            raise click.ClickException(
                "Missing authentication tokens. Please login first."
            )

        return list(metadata) + [
            (ACCESS_TOKEN_KEY, self.access_token),
            (REFRESH_TOKEN_KEY, self.refresh_token),
        ]

    def read_tokens_from_metadata(
        self, metadata: Sequence[tuple[str, str | bytes]]
    ) -> AccountAuthCredentials | None:
        """Read authentication tokens from the provided metadata."""
        metadata_dict = dict(metadata)
        access_token = metadata_dict.get(ACCESS_TOKEN_KEY)
        refresh_token = metadata_dict.get(REFRESH_TOKEN_KEY)

        if isinstance(access_token, str) and isinstance(refresh_token, str):
            return AccountAuthCredentials(
                access_token=access_token,
                refresh_token=refresh_token,
            )

        return None
