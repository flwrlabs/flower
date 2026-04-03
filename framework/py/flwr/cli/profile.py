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
"""Flower command line interface `profile` command."""

import io
import json
from pathlib import Path
from typing import Annotated

import grpc

import typer
from rich.console import Console
from rich.table import Table

from flwr.cli.config_utils import (
    exit_if_no_address,
    load_and_validate,
    process_loaded_project_config,
    validate_federation_in_project_config,
)
from flwr.cli.constant import FEDERATION_CONFIG_HELP_MESSAGE
from flwr.common.constant import CliOutputFormat
from flwr.common.logger import print_json_error, redirect_output, restore_output
from flwr.proto.control_pb2 import GetRunProfileRequest, StreamRunProfileRequest  # pylint: disable=E0611
from flwr.proto.control_pb2_grpc import ControlStub  # pylint: disable=E0611

from .utils import flwr_cli_grpc_exc_handler, init_channel, load_cli_auth_plugin


def profile(
    run_id: Annotated[
        int,
        typer.Argument(help="The Flower run ID to query"),
    ],
    app: Annotated[
        Path,
        typer.Argument(help="Path of the Flower project"),
    ] = Path("."),
    federation: Annotated[
        str | None,
        typer.Argument(help="Name of the federation"),
    ] = None,
    federation_config_overrides: Annotated[
        list[str] | None,
        typer.Option(
            "--federation-config",
            help=FEDERATION_CONFIG_HELP_MESSAGE,
        ),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            case_sensitive=False,
            help="Format output using 'default' view or 'json'",
        ),
    ] = CliOutputFormat.DEFAULT,
    live: Annotated[
        bool,
        typer.Option("--live", help="Stream profile updates while the run is active"),
    ] = False,
) -> None:
    """Get profiling summary for a run."""
    suppress_output = output_format == CliOutputFormat.JSON
    captured_output = io.StringIO()
    try:
        if suppress_output:
            redirect_output(captured_output)

        if not suppress_output:
            typer.secho("Loading project configuration... ", fg=typer.colors.BLUE)
        pyproject_path = app / "pyproject.toml" if app else None
        config, errors, warnings = load_and_validate(pyproject_path, check_module=False)
        config = process_loaded_project_config(config, errors, warnings)
        federation, federation_config = validate_federation_in_project_config(
            federation, config, federation_config_overrides
        )
        exit_if_no_address(federation_config, "profile")

        auth_plugin = load_cli_auth_plugin(app, federation, federation_config)
        channel = init_channel(app, federation_config, auth_plugin)
        try:
            stub = ControlStub(channel)
            if live:
                req = StreamRunProfileRequest(run_id=run_id)
                with flwr_cli_grpc_exc_handler():
                    for res in stub.StreamRunProfile(req):
                        if not res.summary_json:
                            continue
                        summary = json.loads(res.summary_json)
                        restore_output()
                        if output_format == CliOutputFormat.JSON:
                            Console().print_json(json.dumps(summary))
                        else:
                            entries, network_entries = _split_entries(summary)
                            Console().print(_render_table(summary, title="Run Profile Summary", entries=entries))
                            if network_entries:
                                Console().print(
                                    _render_table(
                                        summary,
                                        title="Network Profile",
                                        entries=_rename_network_tasks(network_entries),
                                        include_memory=False,
                                        include_disk=False,
                                    )
                                )
                        if suppress_output:
                            redirect_output(captured_output)
                return
            req = GetRunProfileRequest(run_id=run_id)
            try:
                with flwr_cli_grpc_exc_handler():
                    res = stub.GetRunProfile(req)
            except grpc.RpcError as exc:
                if exc.code() == grpc.StatusCode.NOT_FOUND:
                    restore_output()
                    summary = {"run_id": run_id, "entries": []}
                    if output_format == CliOutputFormat.JSON:
                        Console().print_json(json.dumps(summary))
                    else:
                        Console().print(_to_table(summary))
                    return
                raise
        finally:
            channel.close()

        if not res.summary_json:
            summary = {"run_id": run_id, "entries": []}
            restore_output()
            if output_format == CliOutputFormat.JSON:
                Console().print_json(json.dumps(summary))
            else:
                Console().print(_to_table(summary))
            return

        summary = json.loads(res.summary_json)
        restore_output()
        if output_format == CliOutputFormat.JSON:
            Console().print_json(json.dumps(summary))
        else:
            entries, network_entries = _split_entries(summary)
            Console().print(_render_table(summary, title="Run Profile Summary", entries=entries))
            if network_entries:
                Console().print(
                    _render_table(
                        summary,
                        title="Network Profile",
                        entries=_rename_network_tasks(network_entries),
                        include_memory=False,
                        include_disk=False,
                    )
                )
    except (typer.Exit, Exception) as err:  # pylint: disable=broad-except
        if suppress_output:
            restore_output()
            e_message = captured_output.getvalue()
            print_json_error(e_message, err)
        else:
            typer.secho(
                f"{err}",
                fg=typer.colors.RED,
                bold=True,
                err=True,
            )
    finally:
        if suppress_output:
            restore_output()
        captured_output.close()


def _to_table(summary: dict) -> Table:
    """Format the summary to a rich Table."""
    return _render_table(summary, title="Run Profile Summary")


def _render_table(
    summary: dict,
    *,
    title: str,
    entries: list[dict] | None = None,
    include_memory: bool = True,
    include_disk: bool = True,
) -> Table:
    table_title = title
    if title == "Run Profile Summary":
        total_execution_ms = summary.get("total_execution_ms")
        if isinstance(total_execution_ms, (int, float)):
            table_title = (
                f"{title} (Total execution: {total_execution_ms / 1000.0:.2f}s)"
            )
    table = Table(title=table_title)
    table.add_column("Task", style="white")
    table.add_column("Scope", style="white")
    table.add_column("Round", style="cyan")
    table.add_column("Node", style="magenta")
    table.add_column("Avg (ms)", justify="right")
    table.add_column("Max (ms)", justify="right")
    if include_memory:
        table.add_column("Avg Mem (MB)", justify="right")
        table.add_column("Max Mem (MB)", justify="right")
        table.add_column("Avg ΔMem (MB)", justify="right")
    if include_disk:
        table.add_column("Avg Read (MB)", justify="right")
        table.add_column("Avg Write (MB)", justify="right")
        table.add_column("Disk Src", justify="right")
    table.add_column("Count", justify="right")

    use_entries = entries if entries is not None else summary.get("entries", [])
    for entry in use_entries:
        node_val = entry.get("node_id")
        if node_val is None and entry.get("scope") == "server":
            node_val = "server"
        avg_mem = entry.get("avg_mem_mb")
        max_mem = entry.get("max_mem_mb")
        avg_mem_delta = entry.get("avg_mem_delta_mb")
        avg_read = entry.get("avg_disk_read_mb")
        avg_write = entry.get("avg_disk_write_mb")
        disk_source = entry.get("disk_source")
        table.add_row(
            str(entry.get("task", "")),
            str(entry.get("scope", "")),
            str(entry.get("round", "N/A")),
            str(node_val if node_val is not None else "N/A"),
            f"{entry.get('avg_ms', 0.0):.2f}",
            f"{entry.get('max_ms', 0.0):.2f}",
            *(
                [
                    f"{avg_mem:.2f}" if isinstance(avg_mem, (int, float)) else "-",
                    f"{max_mem:.2f}" if isinstance(max_mem, (int, float)) else "-",
                    f"{avg_mem_delta:.2f}"
                    if isinstance(avg_mem_delta, (int, float))
                    else "-",
                ]
                if include_memory
                else []
            ),
            *(
                [
                    f"{avg_read:.2f}" if isinstance(avg_read, (int, float)) else "-",
                    f"{avg_write:.2f}" if isinstance(avg_write, (int, float)) else "-",
                    str(disk_source) if disk_source else "-",
                ]
                if include_disk
                else []
            ),
            str(entry.get("count", 0)),
        )
    return table


def _split_entries(summary: dict) -> tuple[list[dict], list[dict]]:
    entries = summary.get("entries", [])
    network_entries: list[dict] = []
    other_entries: list[dict] = []
    for entry in entries:
        if entry.get("scope") == "client" and entry.get("task") == "total":
            continue
        if entry.get("scope") == "network":
            network_entries.append(entry)
        else:
            other_entries.append(entry)
    return other_entries, network_entries


def _rename_network_tasks(entries: list[dict]) -> list[dict]:
    renamed: list[dict] = []
    mapping = {
        "network_upstream": "upstream",
        "network_downstream": "downstream",
        "send_and_receive": "combined",
    }
    for entry in entries:
        task = entry.get("task")
        if task in mapping:
            entry = {**entry, "task": mapping[task]}
        renamed.append(entry)
    return renamed
