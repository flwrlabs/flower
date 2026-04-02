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
"""Flower command line interface `federation archive` command."""

from typing import Annotated, Literal

import typer

from flwr.cli.config_migration import migrate
from flwr.cli.flower_config import read_superlink_connection
from flwr.common.constant import CliOutputFormat
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ArchiveFederationRequest,
    ArchiveFederationResponse,
)
from flwr.proto.control_pb2_grpc import ControlStub

from ..utils import (
    cli_output_handler,
    flwr_cli_grpc_exc_handler,
    init_channel_from_connection,
    print_json_to_stdout,
)


def archive(
    ctx: typer.Context,
    federation: Annotated[
        str,
        typer.Argument(help="Name of the federation to archive."),
    ],
    superlink: Annotated[
        str | None,
        typer.Argument(help="Name of the SuperLink connection."),
    ] = None,
    output_format: Annotated[
        Literal["default", "json"],
        typer.Option(
            "--format",
            case_sensitive=False,
            help="Format output using 'default' view or 'json'",
        ),
    ] = CliOutputFormat.DEFAULT,
) -> None:
    """Archive an existing federation."""
    with cli_output_handler(output_format=output_format) as is_json:
        # Migrate legacy usage if any
        migrate(superlink, args=ctx.args)

        # Read superlink connection configuration
        superlink_connection = read_superlink_connection(superlink)
        channel = None

        try:
            channel = init_channel_from_connection(superlink_connection)
            stub = ControlStub(channel)

            request = ArchiveFederationRequest(
                federation_name=federation,
            )
            _archive_federation(
                stub=stub,
                request=request,
                is_json=is_json,
            )

        finally:
            if channel:
                channel.close()


def _archive_federation(  # pylint: disable=W0613
    stub: ControlStub,
    request: ArchiveFederationRequest,
    is_json: bool,
) -> None:
    """Archive a federation."""
    with flwr_cli_grpc_exc_handler():
        _: ArchiveFederationResponse = stub.ArchiveFederation(request)

    if is_json:
        print_json_to_stdout({"success": True, "name": request.federation_name})
    else:
        typer.secho(f"✅ Federation '{request.federation_name}' archived successfully.")
