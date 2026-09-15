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
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    GetRunProfileRequest,
    StreamRunProfileRequest,
)
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
                            Console().print(
                                _render_table(
                                    summary,
                                    title="Run Profile Summary",
                                    entries=entries,
                                )
                            )
                            if network_entries:
                                Console().print(
                                    _render_transport_table(
                                        _summarize_transport_wall_clock(summary)
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
            Console().print(
                _render_table(summary, title="Run Profile Summary", entries=entries)
            )
            if network_entries:
                Console().print(
                    _render_transport_table(_summarize_transport_wall_clock(summary))
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
    include_network: bool = False,
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
    table.add_column("Total (ms)", justify="right")
    table.add_column("Avg (ms)", justify="right")
    table.add_column("Max (ms)", justify="right")
    if include_network:
        table.add_column("Sender", style="white")
        table.add_column("Receiver", style="white")
        table.add_column("Avg Data (MB)", justify="right")
        table.add_column("Total Data (MB)", justify="right")
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
        node_display = entry.get("node_name") or node_val
        avg_mem = entry.get("avg_mem_mb")
        max_mem = entry.get("max_mem_mb")
        avg_mem_delta = entry.get("avg_mem_delta_mb")
        avg_read = entry.get("avg_disk_read_mb")
        avg_write = entry.get("avg_disk_write_mb")
        disk_source = entry.get("disk_source")
        sender_node = entry.get("sender_node_id")
        receiver_node = entry.get("receiver_node_id")
        sender_display = entry.get("sender_node_name") or sender_node
        receiver_display = entry.get("receiver_node_name") or receiver_node
        avg_network = entry.get("avg_network_mb")
        total_network = entry.get("total_network_mb")
        total_ms = entry.get(
            "total_ms", entry.get("avg_ms", 0.0) * entry.get("count", 0)
        )
        round_value = entry.get("round")
        table.add_row(
            str(entry.get("task", "")),
            str(entry.get("scope", "")),
            str(round_value if round_value is not None else "N/A"),
            str(node_display if node_display is not None else "N/A"),
            f"{total_ms:.2f}",
            f"{entry.get('avg_ms', 0.0):.2f}",
            f"{entry.get('max_ms', 0.0):.2f}",
            *(
                [
                    str(sender_display) if sender_display is not None else "-",
                    str(receiver_display) if receiver_display is not None else "-",
                    (
                        f"{avg_network:.2f}"
                        if isinstance(avg_network, (int, float))
                        else "-"
                    ),
                    (
                        f"{total_network:.2f}"
                        if isinstance(total_network, (int, float))
                        else "-"
                    ),
                ]
                if include_network
                else []
            ),
            *(
                [
                    f"{avg_mem:.2f}" if isinstance(avg_mem, (int, float)) else "-",
                    f"{max_mem:.2f}" if isinstance(max_mem, (int, float)) else "-",
                    (
                        f"{avg_mem_delta:.2f}"
                        if isinstance(avg_mem_delta, (int, float))
                        else "-"
                    ),
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
        if entry.get("scope") in {"network", "transport"}:
            network_entries.append(entry)
        elif entry.get("scope") == "server" and entry.get("task") in {
            "network_downstream",
            "network_upstream",
        }:
            task = entry.get("task")
            network_entries.append(
                {
                    **entry,
                    "scope": "transport",
                    "task": (
                        "serverapp_superlink_downstream"
                        if task == "network_downstream"
                        else "serverapp_superlink_upstream"
                    ),
                    "sender_node_id": (
                        "serverapp" if task == "network_downstream" else "superlink"
                    ),
                    "receiver_node_id": (
                        "superlink" if task == "network_downstream" else "serverapp"
                    ),
                }
            )
        else:
            other_entries.append(entry)
    return other_entries, network_entries


def _union_duration_ms(intervals: list[tuple[float, float]]) -> float:
    """Return the wall-clock duration covered by a set of intervals."""
    if not intervals:
        return 0.0
    total_ms = 0.0
    current_start, current_end = sorted(intervals)[0]
    for start, end in sorted(intervals)[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total_ms += current_end - current_start
        current_start, current_end = start, end
    return total_ms + current_end - current_start


def _summarize_transport_wall_clock(summary: dict) -> list[dict]:
    """Summarize non-overlapping communication wall time from raw events."""
    hop_labels = {
        "serverapp_clientapp": "Full path: ServerApp <-> ClientApp",
        "serverapp_superlink": "ServerApp <-> SuperLink",
        "superlink_supernode": "SuperLink <-> SuperNode",
        "supernode_clientapp": "SuperNode <-> ClientApp",
    }
    events = summary.get("events", [])
    has_direct_serverapp_events = any(
        str(event.get("task", "")).startswith("serverapp_superlink_")
        for event in events
    )
    grouped: dict[tuple[str, object, object], dict] = {}
    for event in events:
        task = str(event.get("task", ""))
        if event.get("scope") == "server" and task in {
            "network_downstream",
            "network_upstream",
        }:
            if has_direct_serverapp_events:
                continue
            task = f"serverapp_superlink_{task.removeprefix('network_')}"
        hop = next(
            (
                name
                for name in hop_labels
                if task in {f"{name}_upstream", f"{name}_downstream"}
            ),
            None,
        )
        if hop is None:
            continue
        direction = "downstream" if task.endswith("_downstream") else "upstream"
        node_id = "all" if hop == "serverapp_superlink" else event.get("node_id")
        key = (hop, event.get("round"), node_id)
        group = grouped.setdefault(
            key,
            {
                "hop": hop_labels[hop],
                "hop_key": hop,
                "round": event.get("round"),
                "node_id": node_id,
                "node_name": event.get("node_name"),
                "downstream_intervals": [],
                "upstream_intervals": [],
                "downstream_mb": 0.0,
                "upstream_mb": 0.0,
                "downstream_count": 0,
                "upstream_count": 0,
            },
        )
        timestamp_ms = event.get("timestamp_ms")
        duration_ms = event.get("duration_ms")
        if isinstance(timestamp_ms, (int, float)) and isinstance(
            duration_ms, (int, float)
        ):
            start = float(timestamp_ms)
            group[f"{direction}_intervals"].append(
                (start, start + max(float(duration_ms), 0.0))
            )
        if isinstance(event.get("network_mb"), (int, float)):
            group[f"{direction}_mb"] += float(event["network_mb"])
        group[f"{direction}_count"] += 1

    # Add wall-clock union rows across parallel clients for each boundary.
    aggregates: dict[tuple[str, object], dict] = {}
    for group in grouped.values():
        if group["hop_key"] == "serverapp_superlink":
            continue
        aggregate = aggregates.setdefault(
            (group["hop_key"], group["round"]),
            {
                **group,
                "node_id": "all",
                "node_name": None,
                "downstream_intervals": [],
                "upstream_intervals": [],
                "downstream_mb": 0.0,
                "upstream_mb": 0.0,
                "downstream_count": 0,
                "upstream_count": 0,
            },
        )
        for field in (
            "downstream_intervals",
            "upstream_intervals",
        ):
            aggregate[field].extend(group[field])
        for field in (
            "downstream_mb",
            "upstream_mb",
            "downstream_count",
            "upstream_count",
        ):
            aggregate[field] += group[field]

    groups = list(grouped.values()) + list(aggregates.values())

    result = []
    for group in groups:
        down_intervals = group.pop("downstream_intervals")
        up_intervals = group.pop("upstream_intervals")
        result.append(
            {
                **group,
                "downstream_ms": _union_duration_ms(down_intervals),
                "upstream_ms": _union_duration_ms(up_intervals),
                "combined_ms": _union_duration_ms(down_intervals + up_intervals),
            }
        )
    return sorted(
        result,
        key=lambda row: (
            row["round"] if isinstance(row["round"], int) else -1,
            {
                "serverapp_clientapp": 0,
                "serverapp_superlink": 1,
                "superlink_supernode": 2,
                "supernode_clientapp": 3,
            }.get(row["hop_key"], 99),
            str(row["node_id"]),
        ),
    )


def _render_transport_table(rows: list[dict]) -> Table:
    """Render communication wall-clock time for each measured boundary."""
    table = Table(title="Communication Profile (wall clock per round)")
    table.caption = (
        "Full path measures model availability from ServerApp to ClientApp and the "
        "complete update back to ServerApp. Combined is the union of measured "
        "intervals, so overlap is counted once. Cross-host measurements require "
        "synchronized clocks."
    )
    table.add_column("Boundary", style="white")
    table.add_column("Round", style="cyan")
    table.add_column("Node", style="magenta")
    table.add_column("Down (ms)", justify="right")
    table.add_column("Up (ms)", justify="right")
    table.add_column("Combined (ms)", justify="right")
    table.add_column("Down (MB)", justify="right")
    table.add_column("Up (MB)", justify="right")
    table.add_column("Total (MB)", justify="right")
    table.add_column("Down Msgs", justify="right")
    table.add_column("Up Msgs", justify="right")
    for row in rows:
        node = row.get("node_name") or row.get("node_id") or "N/A"
        down_ms = float(row["downstream_ms"])
        up_ms = float(row["upstream_ms"])
        down_mb = float(row["downstream_mb"])
        up_mb = float(row["upstream_mb"])
        combined_ms = float(row["combined_ms"])
        table.add_row(
            str(row["hop"]),
            str(row["round"] if row["round"] is not None else "N/A"),
            str(node),
            f"{down_ms:.2f}",
            f"{up_ms:.2f}",
            f"{combined_ms:.2f}",
            f"{down_mb:.2f}",
            f"{up_mb:.2f}",
            f"{down_mb + up_mb:.2f}",
            str(row["downstream_count"]),
            str(row["upstream_count"]),
        )
    return table
