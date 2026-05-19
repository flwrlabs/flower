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
"""Flower command line interface `ls` command."""


import json
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from flwr.cli.config_migration import migrate, warn_if_federation_config_overrides
from flwr.cli.constant import FEDERATION_CONFIG_HELP_MESSAGE
from flwr.cli.flower_config import read_superlink_connection
from flwr.common.constant import CliOutputFormat, Status, SubStatus
from flwr.common.serde import run_from_proto
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListRunsRequest,
    ListRunsResponse,
)
from flwr.proto.control_pb2_grpc import ControlStub
from flwr.supercore.utils import humanize_bytes, humanize_duration

from .run_utils import RunRow, format_runs
from .utils import (
    cli_output_handler,
    flwr_cli_grpc_exc_handler,
    init_channel_from_connection,
    print_json_to_stdout,
)


def ls(  # pylint: disable=too-many-locals, too-many-branches, R0913, R0917
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
    run_id: Annotated[
        int | None,
        typer.Option(
            "--run-id",
            help="Specific run ID to display",
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            help="Maximum number of runs to display",
            min=1,
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
    """List the details of one provided run ID or all runs (alias: ls).

    The following details are displayed:

    - **Run ID:** Unique identifier for the run.
    - **Federation:** The federation to which the run belongs.
    - **App:** The App associated with the run (``<APP_ID>==<APP_VERSION>``).
    - **Status:** Current status of the run (pending, starting, running, finished).
    - **Elapsed:** Time elapsed since the run started (``HH:MM:SS``).
    - **Status Changed @:** Timestamp of the most recent status change.

    All timestamps follow ISO 8601, UTC and are formatted as ``YYYY-MM-DD HH:MM:SSZ``.
    """
    with cli_output_handler(output_format=output_format) as is_json:
        # Warn `--federation-config` is ignored
        warn_if_federation_config_overrides(federation_config_overrides)

        # Migrate legacy usage if any
        migrate(superlink, args=ctx.args)

        # Read superlink connection configuration
        superlink_connection = read_superlink_connection(superlink)
        channel = None

        # Check `--limit` is not used together with `--run-id`
        if limit is not None and run_id is not None:
            raise ValueError(
                "The options '--run-id' and '--limit' cannot be used together."
            )

        try:
            channel = init_channel_from_connection(superlink_connection)
            stub = ControlStub(channel)

            # Display information about a specific run ID
            if run_id is not None:
                typer.echo(f"🔍 Displaying information for run ID {run_id}...")
                formatted_runs = _display_one_run(stub, run_id)
            # By default, list all runs
            else:
                typer.echo("📄 Listing all runs...")
                formatted_runs = _list_runs(stub, limit)

            if is_json:
                print_json_to_stdout(_to_json(formatted_runs))
            else:
                if run_id is not None:
                    Console().print(_to_detail_table(formatted_runs[0]))
                else:
                    Console().print(_to_table(formatted_runs))
        finally:
            if channel:
                channel.close()


def _get_status_style(status_text: str) -> str:
    """Determine the display style/color for a status.

    Parameters
    ----------
    status_text : str
        The status text to determine color for.

    Returns
    -------
    str
        Color name for rich console styling (e.g., 'green', 'red', 'blue').
    """
    status = status_text.lower()
    sub_status = status_text.rsplit(":", maxsplit=1)[-1]

    if sub_status == SubStatus.COMPLETED:  # finished:completed
        return "green"
    if sub_status == SubStatus.FAILED:  # finished:failed
        return "red"
    if sub_status == SubStatus.STOPPED:  # finished:stopped
        return "yellow"
    if status in (Status.STARTING, Status.RUNNING):  # starting, running
        return "blue"
    return "bright_black"  # pending


def _to_table(run_list: list[RunRow]) -> Table:
    """Format the provided run list to a rich Table.

    Parameters
    ----------
    run_list : list[RunRow]
        List of run information to display.

    Returns
    -------
    Table
        Rich Table object with formatted run information.
    """
    table = Table(header_style="bold cyan", show_lines=True)

    # Add columns
    table.add_column(Text("Run ID", justify="center"), no_wrap=True)
    table.add_column(Text("Federation", justify="center"))
    table.add_column(Text("App", justify="center"))
    table.add_column(Text("Status", justify="center"))
    table.add_column(Text("Elapsed", justify="center"), style="blue")
    table.add_column(Text("Status Changed @", justify="center"))

    for row in run_list:
        status_style = _get_status_style(row.status_text)

        # Use the most recent timestamp
        if row.finished_at != "N/A":
            status_changed_at = row.finished_at
        elif row.running_at != "N/A":
            status_changed_at = row.running_at
        elif row.starting_at != "N/A":
            status_changed_at = row.starting_at
        else:
            status_changed_at = row.pending_at

        formatted_row = (
            f"[bold]{row.run_id}[/bold]",
            row.federation,
            f"@{row.fab_id}=={row.fab_version}",
            f"[{status_style}]{row.status_text}[/{status_style}]",
            humanize_duration(row.elapsed),
            status_changed_at,
        )
        table.add_row(*formatted_row)

    return table


def _to_detail_table(run: RunRow) -> Table:
    """Format a single run's details in a vertical table layout.

    Parameters
    ----------
    run : RunRow
        The run information to display.

    Returns
    -------
    Table
        Rich Table object with detailed run information in vertical format.
    """
    status_style = _get_status_style(run.status_text)

    # Create vertical table with field names on the left
    table = Table(show_header=False, show_lines=False)
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value")

    # Add rows with all details
    table.add_row("Run ID", f"[bold]{run.run_id}[/bold]")
    table.add_row("Federation", run.federation)
    table.add_row("App", f"@{run.fab_id}=={run.fab_version}")
    table.add_row("FAB Hash", f"{run.fab_hash[:8]}...{run.fab_hash[-8:]}")
    table.add_row("Status", f"[{status_style}]{run.status_text}[/{status_style}]")
    table.add_row("Elapsed", f"[blue]{humanize_duration(run.elapsed)}[/blue]")
    table.add_row("Pending At", run.pending_at)
    table.add_row("Starting At", run.starting_at)
    table.add_row("Running At", run.running_at)
    table.add_row("Finished At", run.finished_at)
    table.add_row(
        "Network traffic (inbound)",
        f"[blue]{humanize_bytes(run.network_traffic_inbound)}[/blue]",
    )
    table.add_row(
        "Network traffic (outbound)",
        f"[blue]{humanize_bytes(run.network_traffic_outbound)}[/blue]",
    )
    table.add_row(
        "Network Traffic (total)",
        "[blue]"
        f"{humanize_bytes(run.network_traffic_inbound + run.network_traffic_outbound)}"
        "[/blue]",
    )
    table.add_row(
        "Compute Time (ServerApp)",
        f"[blue]{humanize_duration(run.compute_time_serverapp)}[/blue]",
    )
    table.add_row(
        "Compute Time (ClientApp)",
        f"[blue]{humanize_duration(run.compute_time_clientapp)}[/blue]",
    )
    table.add_row(
        "Compute Time (total)",
        "[blue]"
        f"{humanize_duration(run.compute_time_serverapp + run.compute_time_clientapp)}"
        "[/blue]",
    )

    return table


def _to_json(run_list: list[RunRow]) -> str:
    """Format run status list to a JSON formatted string.

    Parameters
    ----------
    run_list : list[RunRow]
        List of run information to serialize.

    Returns
    -------
    str
        JSON string containing formatted run information.
    """
    runs_list = []
    for row in run_list:
        runs_list.append(
            {
                "run-id": f"{row.run_id}",
                "federation": row.federation,
                "fab-id": row.fab_id,
                "fab-name": row.fab_id.split("/")[-1],
                "fab-version": row.fab_version,
                "fab-hash": row.fab_hash,
                "status": row.status_text,
                "elapsed": row.elapsed,
                "pending-at": row.pending_at,
                "starting-at": row.starting_at,
                "running-at": row.running_at,
                "finished-at": row.finished_at,
                "network-traffic": {
                    "inbound-bytes": row.network_traffic_inbound,
                    "outbound-bytes": row.network_traffic_outbound,
                    "total-bytes": row.network_traffic_inbound
                    + row.network_traffic_outbound,
                },
                "compute-time": {
                    "serverapp-seconds": row.compute_time_serverapp,
                    "clientapp-seconds": row.compute_time_clientapp,
                    "total-seconds": row.compute_time_serverapp
                    + row.compute_time_clientapp,
                },
            }
        )

    return json.dumps({"success": True, "runs": runs_list})


def _list_runs(stub: ControlStub, limit: int | None = None) -> list[RunRow]:
    """List all runs.

    Parameters
    ----------
    stub : ControlStub
        The gRPC stub for Control API communication.

    Returns
    -------
    list[RunRow]
        List of formatted run information for all runs.
    """
    with flwr_cli_grpc_exc_handler():
        res: ListRunsResponse = stub.ListRuns(ListRunsRequest(limit=limit))
    runs = [run_from_proto(proto) for proto in res.run_dict.values()]

    return format_runs(runs, res.now)


def _display_one_run(stub: ControlStub, run_id: int) -> list[RunRow]:
    """Display information about a specific run.

    Parameters
    ----------
    stub : ControlStub
        The gRPC stub for Control API communication.
    run_id : int
        The unique identifier of the run to display.

    Returns
    -------
    list[RunRow]
        List containing the formatted run information (single item).

    Raises
    ------
    ValueError
        If the run_id is not found.
    """
    with flwr_cli_grpc_exc_handler():
        res: ListRunsResponse = stub.ListRuns(ListRunsRequest(run_id=run_id))
    if not res.run_dict:
        # This won't be reached as an gRPC error is raised if run_id is invalid
        raise ValueError(f"Run ID {run_id} not found")

    runs = [run_from_proto(proto) for proto in res.run_dict.values()]
    return format_runs(runs, res.now)
