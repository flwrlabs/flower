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
"""Flower command line interface `login` command."""


from typing import Annotated, cast

import click
import typer

from flwr.cli.auth_plugin import LoginError, NoOpCliAuthPlugin
from flwr.cli.constant import FEDERATION_CONFIG_HELP_MESSAGE
from flwr.cli.utils import init_channel_from_connection
from flwr.common.typing import AccountAuthLoginDetails
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    GetLoginDetailsRequest,
    GetLoginDetailsResponse,
)
from flwr.proto.control_pb2_grpc import ControlStub

from ..config_migration import migrate, warn_if_federation_config_overrides
from ..flower_config import read_superlink_connection
from ..utils import flwr_cli_grpc_exc_handler, load_cli_auth_plugin_from_connection


def login(
    ctx: typer.Context,
    superlink: Annotated[
        str | None,
        typer.Argument(help="Name of the SuperLink connection."),
    ] = None,
    federation_config_overrides: Annotated[
        list[str] | None,
        typer.Option(
            "--federation-config",
            help=FEDERATION_CONFIG_HELP_MESSAGE,
            hidden=True,
        ),
    ] = None,
) -> None:
    """Login to Flower SuperLink."""
    # Warn `--federation-config` is ignored
    warn_if_federation_config_overrides(federation_config_overrides)

    # Migrate legacy usage if any
    migrate(superlink, args=ctx.args)

    # Read superlink connection configuration
    superlink_connection = read_superlink_connection(superlink)
    superlink = superlink_connection.name

    # Check if insecure flag is set to `True`
    if superlink_connection.insecure:
        raise click.ClickException(
            "`flwr login` requires TLS to be enabled. `insecure` must NOT be set to "
            "`true` in the federation configuration."
        )

    channel = init_channel_from_connection(superlink_connection, NoOpCliAuthPlugin())
    stub = ControlStub(channel)

    login_request = GetLoginDetailsRequest()
    with flwr_cli_grpc_exc_handler():
        login_response: GetLoginDetailsResponse = stub.GetLoginDetails(login_request)

    # Get the auth plugin
    authn_plugin = load_cli_auth_plugin_from_connection(
        cast(str, superlink_connection.address), login_response.authn_type
    )

    # Login
    details = AccountAuthLoginDetails(
        authn_type=login_response.authn_type,
        device_code=login_response.device_code,
        verification_uri_complete=login_response.verification_uri_complete,
        expires_in=login_response.expires_in,
        interval=login_response.interval,
    )
    try:
        with flwr_cli_grpc_exc_handler():
            credentials = authn_plugin.login(details, stub)
        typer.secho(
            "✅ Login successful.",
            fg=typer.colors.GREEN,
            bold=False,
        )
    except LoginError as e:
        raise click.ClickException(f"Login failed: {e.message}") from None

    # Store the tokens
    authn_plugin.store_tokens(credentials)
