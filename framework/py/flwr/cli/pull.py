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
"""Flower command line interface `pull` command."""


from typing import Annotated

import click
import typer

from flwr.cli.config_migration import migrate, warn_if_federation_config_overrides
from flwr.cli.constant import FEDERATION_CONFIG_HELP_MESSAGE
from flwr.cli.flower_config import read_superlink_connection
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    PullArtifactsRequest,
    PullArtifactsResponse,
)
from flwr.proto.control_pb2_grpc import ControlStub

from .utils import flwr_cli_grpc_exc_handler, init_channel_from_connection


def pull(  # pylint: disable=R0914
    ctx: typer.Context,
    run_id: Annotated[
        int,
        typer.Argument(help="Run ID to pull artifacts from."),
    ],
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
    """Pull artifacts from a Flower run.

    Retrieve a download URL for artifacts generated during a completed Flower run. The
    artifacts can then be downloaded from the provided URL.
    """
    # Warn `--federation-config` is ignored
    warn_if_federation_config_overrides(federation_config_overrides)

    # Migrate legacy usage if any
    migrate(superlink, args=ctx.args)

    # Read superlink connection configuration
    superlink_connection = read_superlink_connection(superlink)
    channel = None
    try:
        channel = init_channel_from_connection(superlink_connection)
        stub = ControlStub(channel)
        with flwr_cli_grpc_exc_handler():
            res: PullArtifactsResponse = stub.PullArtifacts(
                PullArtifactsRequest(run_id=run_id)
            )

        if not res.url:
            raise click.ClickException(
                f"A download URL for artifacts from run {run_id} couldn't be obtained."
            )

        typer.secho(
            f"✅ Artifacts for run {run_id} can be downloaded from: {res.url}",
            fg=typer.colors.GREEN,
        )
    finally:
        if channel:
            channel.close()
