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
"""Flower command line interface `federation list` command."""


from typing import Annotated, Any, Literal

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from flwr.cli.config_migration import migrate
from flwr.cli.flower_config import read_superlink_connection
from flwr.cli.ls import _get_status_style
from flwr.common.constant import NOOP_ACCOUNT_NAME, CliOutputFormat
from flwr.common.serde import run_from_proto
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListFederationsRequest,
    ListFederationsResponse,
    ShowFederationRequest,
    ShowFederationResponse,
)
from flwr.proto.control_pb2_grpc import ControlStub
from flwr.proto.federation_config_pb2 import SimulationConfig  # pylint: disable=E0611
from flwr.proto.federation_pb2 import Federation, Member  # pylint: disable=E0611
from flwr.proto.node_pb2 import NodeInfo  # pylint: disable=E0611
from flwr.supercore.utils import humanize_duration, simulation_config_to_json

from ..run_utils import RunRow, format_runs
from ..utils import (
    cli_output_handler,
    flwr_cli_grpc_exc_handler,
    init_channel_from_connection,
    print_json_to_stdout,
)


def ls(  # pylint: disable=R0914, R0913, R0917, R0912
    ctx: typer.Context,
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
    federation: Annotated[
        str | None,
        typer.Option(
            "--federation",
            case_sensitive=False,
            help="Name of the federation to display",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Show additional details, including archived federations.",
        ),
    ] = False,
) -> None:
    """List available federations or details of a specific federation (alias: ls)."""
    with cli_output_handler(output_format=output_format) as is_json:
        # Migrate legacy usage if any
        migrate(superlink, args=ctx.args)

        # Read superlink connection configuration
        superlink_connection = read_superlink_connection(superlink)
        channel = None

        try:
            channel = init_channel_from_connection(superlink_connection)
            stub = ControlStub(channel)

            if federation:
                # Show specific federation details
                members, nodes, runs, archived, simulation, config = _show_federation(
                    stub, federation
                )
                archived_str = (
                    "[bold yellow]ARCHIVED[/bold yellow] " if archived else ""
                )
                Console().print(
                    f"📄 Showing {archived_str}federation "
                    f"[bold]'{federation}'[/bold]...\n",
                    highlight=False,
                )
                if is_json:
                    print_json_to_stdout(
                        _to_json(
                            members=members,
                            nodes=nodes,
                            runs=runs,
                            archived=archived,
                            simulation=simulation,
                            config=config,
                        )
                    )
                else:

                    Console().print(_to_members_table(members))
                    if simulation:
                        Console().print(_to_simulation_config_table(config))
                    else:
                        Console().print(_to_nodes_table(nodes, archived))
                    Console().print(_to_runs_table(runs))
            else:
                # List federations
                typer.echo("📄 Listing federations...")
                federations = _list_federations(stub)

                active = [f for f in federations if not f.archived]
                archived_feds = [f for f in federations if f.archived]

                if is_json:
                    print_json_to_stdout(_to_json(federations=federations))
                else:
                    # If verbose, show archived federations after active ones
                    shown = active + archived_feds if verbose else active
                    Console().print(_to_table(shown))
        finally:
            if channel:
                channel.close()


def _list_federations(stub: ControlStub) -> list[Federation]:
    """List all federations."""
    with flwr_cli_grpc_exc_handler():
        res: ListFederationsResponse = stub.ListFederations(ListFederationsRequest())

    return list(res.federations)


def _to_table(federations: list[Federation]) -> Table:
    """Format the provided federations list to a rich Table."""
    table = Table(header_style="bold cyan", show_lines=True)

    # Add columns
    table.add_column(
        Text("Federation", justify="center"), style="bright_black", no_wrap=True
    )
    table.add_column(
        Text("Description", justify="center"), style="bright_black", no_wrap=True
    )
    table.add_column(
        Text("Runtime", justify="center"), style="bright_black", no_wrap=True
    )
    table.add_column(
        Text("Status", justify="center"), style="bright_black", no_wrap=True
    )

    for federation in federations:
        if federation.archived:
            style = "dim"
            status = "archived"
        else:
            style = ""
            status = "[green]active[/green]"
        runtime = "simulation" if federation.simulation else "deployment"
        table.add_row(
            federation.name,
            federation.description,
            runtime,
            status,
            style=style,
        )

    return table


def _to_json(  # pylint: disable=R0913,R0917
    federations: list[Federation] | None = None,
    members: list[Member] | None = None,
    nodes: list[NodeInfo] | None = None,
    runs: list[RunRow] | None = None,
    archived: bool | None = None,
    simulation: bool | None = None,
    config: SimulationConfig | None = None,
) -> dict[str, Any]:
    """Format the provided federations list to JSON serializable format."""
    if federations is not None:
        return {
            "federations": [
                {
                    "name": federation.name,
                    "description": federation.description,
                    "archived": federation.archived,
                    "simulation": federation.simulation,
                }
                for federation in federations
            ]
        }

    if members is None or nodes is None or runs is None:
        return {"federation": {}}

    members_list: list[dict[str, Any]] = []
    nodes_list: list[dict[str, Any]] = []
    runs_list: list[dict[str, Any]] = []

    for member in members:
        members_list.append({"member_id": member.account.name, "role": member.role})

    for node in nodes:
        node_json: dict[str, Any] = {
            "node_id": f"{node.node_id}",
            "owner": node.owner_name,
        }
        if not archived:
            node_json["status"] = node.status
        nodes_list.append(node_json)

    for run in runs:
        runs_list.append(
            {
                "run_id": f"{run.run_id}",
                "app": f"@{run.fab_id}=={run.fab_version}",
                "status": run.status_text,
                "elapsed": run.elapsed,
            }
        )

    return {
        "federation": {
            "members": members_list,
            "nodes": nodes_list,
            "runs": runs_list,
            "archived": archived,
            "simulation": simulation,
            "simulation-config": simulation_config_to_json(config) if config else None,
        }
    }


def _show_federation(
    stub: ControlStub, federation: str
) -> tuple[list[Member], list[NodeInfo], list[RunRow], bool, bool, SimulationConfig]:
    """Show federation details.

    Parameters
    ----------
    stub : ControlStub
        The gRPC stub for Control API communication.
    federation : str
        Name of the federation to show.

    Returns
    -------
    tuple[list[Member], list[NodeInfo], list[RunRow], bool, bool, SimulationConfig]
        A tuple containing (members, nodes, runs, archived, simulation, config).
    """
    with flwr_cli_grpc_exc_handler():
        res: ShowFederationResponse = stub.ShowFederation(
            ShowFederationRequest(federation_name=federation)
        )

    fed_proto = res.federation
    runs = [run_from_proto(run_proto) for run_proto in fed_proto.runs]
    formatted_runs = format_runs(runs, res.now)

    return (
        list(fed_proto.members),
        list(fed_proto.nodes),
        formatted_runs,
        fed_proto.archived,
        fed_proto.simulation,
        fed_proto.config,
    )


def _to_members_table(members: list[Member]) -> Table:
    """Format the provided list of federation members as a rich Table.

    Parameters
    ----------
    members : list[Member]
        List of Member proto objects.

    Returns
    -------
    Table
        Rich Table object with formatted member information.
    """
    table = Table(title="Federation Members", header_style="bold cyan", show_lines=True)

    table.add_column(Text("Account Name", justify="center"), no_wrap=True)
    table.add_column(Text("Role", justify="center"), no_wrap=True)

    for member in members:
        table.add_row(member.account.name, member.role)

    return table


def _to_nodes_table(nodes: list[NodeInfo], archived: bool) -> Table:
    """Format the provided list of federation nodes as a rich Table.

    Parameters
    ----------
    nodes : list[NodeInfo]
        List of NodeInfo objects containing node details.

    archived : bool
        Whether the federation is archived. If True, the status
        column for nodes will not be rendered.

    Returns
    -------
    Table
        Rich Table object with formatted node information.

    Raises
    ------
    ValueError
        If an unexpected node status is encountered.
    """
    table = Table(
        title="SuperNodes in the Federation", header_style="bold cyan", show_lines=True
    )

    # Add columns
    table.add_column(
        Text("Node ID", justify="center"), style="bright_black", no_wrap=True
    )
    table.add_column(Text("Owner", justify="center"))
    if not archived:
        table.add_column(Text("Status", justify="center"))

    for row in nodes:
        owner_name = row.owner_name
        status = row.status

        if status == "online":
            status_style = "green"
        elif status == "offline":
            status_style = "bright_yellow"
        elif status == "unregistered":
            continue
        elif status == "registered":
            status_style = "blue"
        else:
            raise ValueError(f"Unexpected node status '{status}'")

        formatted_row: list[str] = [
            f"[bold]{row.node_id}[/bold]",
            (
                f"{owner_name}"
                if owner_name != NOOP_ACCOUNT_NAME
                else f"[dim]{owner_name}[/dim]"
            ),
        ]
        if not archived:
            formatted_row.append(f"[{status_style}]{status}[/{status_style}]")
        table.add_row(*formatted_row)

    return table


def _to_simulation_config_table(config: SimulationConfig) -> Table:
    """Format the simulation configuration as a rich Table."""
    table = Table(
        title="Simulation Configuration",
        header_style="bold cyan",
        show_lines=True,
    )

    table.add_column(Text("Setting", justify="center"), style="bright_black")
    table.add_column(Text("Key", justify="center"), style="bright_black")
    table.add_column(Text("Value"), justify="right")

    rows = [
        ("Number of Simulated SuperNodes", "num_supernodes"),
        ("ClientApp Resources (CPUs)", "client_resources_num_cpus"),
        ("ClientApp Resources (GPUs)", "client_resources_num_gpus"),
        ("Backend Name", "backend"),
    ]

    for name, key in rows:
        table.add_row(
            name,
            Text(key.replace("_", "-"), style="bold magenta"),
            str(getattr(config, key)),
        )

    return table


def _to_runs_table(run_list: list[RunRow]) -> Table:
    """Format the provided list of federation runs as a rich Table.

    Parameters
    ----------
    run_list : list[RunRow]
        List of RunRow objects containing run details.

    Returns
    -------
    Table
        Rich Table object with formatted run information.
    """
    table = Table(
        title="Runs in the Federation", header_style="bold cyan", show_lines=True
    )

    # Add columns
    table.add_column(Text("Run ID", justify="center"), no_wrap=True)
    table.add_column(Text("App", justify="center"))
    table.add_column(Text("Status", justify="center"))
    table.add_column(Text("Elapsed", justify="center"), style="blue")

    for row in run_list:
        status_style = _get_status_style(row.status_text)

        formatted_row = (
            f"[bold]{row.run_id}[/bold]",
            f"@{row.fab_id}=={row.fab_version}",
            f"[{status_style}]{row.status_text}[/{status_style}]",
            f"{humanize_duration(row.elapsed)}",
        )
        table.add_row(*formatted_row)

    return table
