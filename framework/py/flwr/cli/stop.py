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
"""Flower command line interface `stop` command."""


from typing import Annotated, Literal

import click
import typer

from flwr.cli.config_migration import migrate, warn_if_federation_config_overrides
from flwr.cli.constant import FEDERATION_CONFIG_HELP_MESSAGE
from flwr.cli.flower_config import read_superlink_connection
from flwr.common.constant import CliOutputFormat, Status
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListRunsRequest,
    ListRunsResponse,
    StopRunRequest,
    StopRunResponse,
)
from flwr.supercore.control import ControlHttpClient

from .utils import (
    cli_output_handler,
    flwr_cli_exc_handler,
    init_http_client_from_connection,
    print_json_to_stdout,
)

_RUNS_PAGE_SIZE = 20
_ACTIVE_STATUSES = {Status.PENDING, Status.STARTING, Status.RUNNING}


def stop(  # pylint: disable=R0914
    ctx: typer.Context,
    run_id: Annotated[
        str,
        typer.Argument(help="The Flower run ID, or 'latest' or 'all'"),
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
    output_format: Annotated[
        Literal["default", "json"],
        typer.Option(
            "--format",
            case_sensitive=False,
            help="Format output using 'default' view or 'json'",
        ),
    ] = CliOutputFormat.DEFAULT,
) -> None:
    """Stop a Flower run by ID, or use 'latest' or 'all' for active runs."""
    with cli_output_handler(output_format=output_format) as is_json:
        warn_if_federation_config_overrides(federation_config_overrides)
        migrate(superlink, args=ctx.args)
        superlink_connection = read_superlink_connection(superlink)
        control_client = None
        try:
            control_client = init_http_client_from_connection(superlink_connection)
            if run_id.isdecimal():
                run_ids = [int(run_id)]
            elif run_id == "latest":
                run_ids = _list_run_ids(control_client, latest_only=True)
                if not run_ids:
                    raise click.ClickException("No runs found.")
            elif run_id == "all":
                run_ids = _list_run_ids(control_client, active_only=True)
                if not run_ids:
                    raise click.ClickException("No active runs found.")
            else:
                raise click.BadParameter(
                    "Run ID must be an integer, 'latest', or 'all'.",
                    param_hint="RUN_ID",
                )

            for selected_run_id in run_ids:
                typer.secho(
                    f"✋ Stopping run ID {selected_run_id}...",
                    fg=typer.colors.GREEN,
                )
                _stop_run(
                    stub=control_client,
                    run_id=selected_run_id,
                    is_json=is_json and len(run_ids) == 1,
                )
            if is_json and len(run_ids) > 1:
                print_json_to_stdout(
                    {
                        "success": True,
                        "run-ids": [str(selected_run_id) for selected_run_id in run_ids],
                    }
                )
        finally:
            if control_client:
                control_client.close()


def _list_run_ids(
    stub: ControlHttpClient,
    *,
    latest_only: bool = False,
    active_only: bool = False,
) -> list[int]:
    """Return selected run IDs, paging through the complete run history when needed."""
    run_ids: list[int] = []
    skip = 0
    while True:
        with flwr_cli_exc_handler():
            response: ListRunsResponse = stub.ListRuns(
                request=ListRunsRequest(limit=_RUNS_PAGE_SIZE, skip=skip)
            )
        page = sorted(
            response.run_dict.values(),
            key=lambda run: run.pending_at,
            reverse=True,
        )
        if latest_only:
            return [page[0].run_id] if page else []
        for run in page:
            if not active_only or run.status.status in _ACTIVE_STATUSES:
                run_ids.append(run.run_id)
        if len(page) < _RUNS_PAGE_SIZE:
            break
        skip += len(page)
    return run_ids


def _stop_run(stub: ControlHttpClient, run_id: int, is_json: bool) -> None:
    """Stop a run and display the result."""
    with flwr_cli_exc_handler():
        response: StopRunResponse = stub.StopRun(request=StopRunRequest(run_id=run_id))
    if response.success:
        typer.secho(f"✅ Run {run_id} successfully stopped.", fg=typer.colors.GREEN)
        if is_json:
            print_json_to_stdout({"success": True, "run-id": f"{run_id}"})
    else:
        raise click.ClickException(f"Run {run_id} couldn't be stopped.")
