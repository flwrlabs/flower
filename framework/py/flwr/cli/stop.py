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


from typing import Annotated, Literal, cast

import click
import grpc
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
from flwr.proto.control_pb2_grpc import ControlStub
from flwr.supercore.error import ApiErrorCode, FlowerError

from .utils import (
    cli_output_handler,
    flwr_cli_grpc_exc_handler,
    init_channel_from_connection,
    print_json_to_stdout,
)

RunSelector = Literal["latest", "all"]


def stop(  # pylint: disable=R0914
    ctx: typer.Context,
    run_id: Annotated[  # pylint: disable=unused-argument
        str,
        typer.Argument(help="The Flower run ID to stop, or 'latest' or 'all'"),
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
    """Stop a Flower run.

    This command stops a running Flower App execution by sending a stop request to the
    SuperLink via the Control API.
    """
    with cli_output_handler(output_format=output_format) as is_json:
        parsed_run_id = _parse_run_id(run_id)

        # Warn `--federation-config` is ignored
        warn_if_federation_config_overrides(federation_config_overrides)

        migrate(superlink, args=ctx.args)

        # Read superlink connection configuration
        superlink_connection = read_superlink_connection(superlink)
        channel = None

        try:
            channel = init_channel_from_connection(superlink_connection)
            stub = ControlStub(channel)  # pylint: disable=unused-variable # noqa: F841

            run_ids = _resolve_run_ids(stub, parsed_run_id)
            selector = parsed_run_id if isinstance(parsed_run_id, str) else None
            _stop_runs(stub, run_ids, is_json, selector)

        finally:
            if channel:
                channel.close()


def _parse_run_id(run_id: str) -> int | RunSelector:
    """Parse a numeric run ID or supported selector."""
    selector = run_id.lower()
    if selector == "latest":
        return "latest"
    if selector == "all":
        return "all"
    try:
        resolved_run_id = int(run_id)
    except ValueError:
        raise click.ClickException(
            "RUN_ID must be an integer, 'latest', or 'all'."
        ) from None
    if resolved_run_id < 0:
        raise click.ClickException("RUN_ID must be a non-negative integer.")
    return resolved_run_id


def _resolve_run_ids(stub: ControlStub, run_id: int | RunSelector) -> list[int]:
    """Resolve a parsed run ID or selector to active run IDs."""
    if isinstance(run_id, int):
        return [run_id]

    with flwr_cli_grpc_exc_handler():
        response: ListRunsResponse = stub.ListRuns(ListRunsRequest())
    active_runs = sorted(
        (
            run
            for run in response.run_dict.values()
            if run.status.status != Status.FINISHED
        ),
        key=lambda run: run.pending_at,
        reverse=True,
    )
    if not active_runs:
        raise click.ClickException("No active runs found.")

    if run_id == "latest":
        return [active_runs[0].run_id]
    return [run.run_id for run in active_runs]


def _stop_runs(
    stub: ControlStub,
    run_ids: list[int],
    is_json: bool,
    selector: RunSelector | None,
) -> None:
    """Stop resolved run IDs and display the result."""
    stop_all = selector == "all"
    failures = []
    for run_id in run_ids:
        typer.secho(f"✋ Stopping run ID {run_id}...", fg=typer.colors.GREEN)
        try:
            _stop_run(
                stub=stub,
                run_id=run_id,
                is_json=is_json and not stop_all,
                ignore_finished=selector is not None,
            )
        except click.ClickException as err:
            if not stop_all:
                raise
            failures.append(f"Run {run_id}: {err.format_message()}")

    if failures:
        raise click.ClickException("Failed to stop all runs:\n" + "\n".join(failures))

    if is_json and stop_all:
        print_json_to_stdout(
            {
                "success": True,
                "run-ids": [str(run_id) for run_id in run_ids],
            }
        )


def _stop_run(
    stub: ControlStub,
    run_id: int,
    is_json: bool,
    ignore_finished: bool = False,
) -> None:
    """Stop a run and display the result.

    Parameters
    ----------
    stub : ControlStub
        The gRPC stub for Control API communication.
    run_id : int
        The unique identifier of the run to stop.
    is_json : bool
        Whether JSON output format is requested.
    ignore_finished : bool (default: False)
        Whether an already-finished run should be treated as successfully stopped.
    """

    def raise_if_already_finished(error: grpc.RpcError) -> None:
        details = cast(str, error.details())  # pylint: disable=E1101
        flower_error = FlowerError.from_json(details)
        if (
            ignore_finished
            and flower_error is not None
            and flower_error.code == ApiErrorCode.RUN_ALREADY_FINISHED
        ):
            raise _RunAlreadyFinishedError

    try:
        with flwr_cli_grpc_exc_handler(custom_handler=raise_if_already_finished):
            response: StopRunResponse = stub.StopRun(
                request=StopRunRequest(run_id=run_id)
            )
    except _RunAlreadyFinishedError:
        _print_already_finished(run_id, is_json)
        return
    if response.success:
        typer.secho(f"✅ Run {run_id} successfully stopped.", fg=typer.colors.GREEN)
        if is_json:
            print_json_to_stdout(
                {
                    "success": True,
                    "run-id": f"{run_id}",
                }
            )
    elif ignore_finished and _is_run_finished(stub, run_id):
        _print_already_finished(run_id, is_json)
    else:
        raise click.ClickException(f"Run {run_id} couldn't be stopped.")


def _is_run_finished(stub: ControlStub, run_id: int) -> bool:
    """Check whether a run finished during a stop request."""
    with flwr_cli_grpc_exc_handler():
        response: ListRunsResponse = stub.ListRuns(ListRunsRequest(run_id=run_id))
    run = response.run_dict.get(run_id)
    return run is not None and run.status.status == Status.FINISHED


def _print_already_finished(run_id: int, is_json: bool) -> None:
    """Display that a run already reached the requested finished state."""
    typer.secho(f"ℹ️ Run {run_id} already finished.", fg=typer.colors.YELLOW)
    if is_json:
        print_json_to_stdout({"success": True, "run-id": f"{run_id}"})


class _RunAlreadyFinishedError(Exception):
    """Signal that a batch stop target has already finished."""
