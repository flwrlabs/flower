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
from flwr.common.constant import CliOutputFormat
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


def stop(  # pylint: disable=R0914
    ctx: typer.Context,
    run_id: Annotated[  # pylint: disable=unused-argument
        str,
        typer.Argument(help="The Flower run ID to stop. `all` will stop all runs, and `latest` will stop the most recently submitted run."),
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
        # Warn `--federation-config` is ignored
        warn_if_federation_config_overrides(federation_config_overrides)

        migrate(superlink, args=ctx.args)

        # Read superlink connection configuration
        superlink_connection = read_superlink_connection(superlink)
        control_client = None

        try:
            control_client = init_http_client_from_connection(superlink_connection)

            run_ids_to_stop: list[int] = []
            if run_id == "all":
                with flwr_cli_exc_handler():
                    received: int = 0
                    while True:
                        res: ListRunsResponse = control_client.ListRuns(ListRunsRequest(skip=received))
                        if not res.run_dict:
                            break
                        run_ids_to_stop.extend([
                            proto.run_id
                            for proto in res.run_dict.values()
                            if proto.status.status != "finished"
                        ])
                        received += len(res.run_dict)
                        
                # No runs found to stop when attempting to stop all jobs is not
                # an error per se, so don't give it an error message.
                if len(run_ids_to_stop) == 0:
                    typer.secho("No runs found to stop.", fg=typer.colors.GREEN)
                        
            elif run_id == "latest":
                with flwr_cli_exc_handler():
                    res: ListRunsResponse = control_client.ListRuns(  # type: ignore[no-redef]
                        ListRunsRequest(limit=1)
                    )
                run_ids_to_stop = [proto.run_id for proto in res.run_dict.values() if proto.status.status != "finished"]
                
                # If the user is trying to stop the latest run but it's not running
                # in the first place, that more likely means they meant to stop something
                # else - hence, it is an error here.
                if len(run_ids_to_stop) == 0:
                    raise click.ClickException(f"The latest run is not currently running.")
            else:
                try:
                    run_ids_to_stop = [int(run_id)]
                except ValueError:
                    raise click.ClickException(f"The provided run ID \"{run_id}\" is not a valid run ID or selector.")

            stop_responses: list[dict] = []
            for run_id_int in run_ids_to_stop:
                typer.secho(
                    f"✋ Stopping run ID {run_id_int}...", fg=typer.colors.GREEN
                )
                stop_success: bool = _stop_run(stub=control_client, run_id=run_id_int, is_json=is_json)
                stop_responses.append({
                    "success": stop_success,
                    "run-id": f"{run_id_int}"
                })

            if is_json:
                if len(stop_responses) == 1:
                    print_json_to_stdout(stop_responses[0])
                else:
                    print_json_to_stdout(stop_responses)
        finally:
            if control_client:
                control_client.close()


def _stop_run(stub: ControlHttpClient, run_id: int, is_json: bool) -> bool:
    """Stop a run and display the result.

    Parameters
    ----------
    stub : ControlHttpClient
        The HTTP client for Control API communication.
    run_id : int
        The unique identifier of the run to stop.
    is_json : bool
        Whether JSON output format is requested.
    """
    # TODO: Detect a run that is already stopped, and send a different return value
    # for it.
    with flwr_cli_exc_handler():
        response: StopRunResponse = stub.StopRun(request=StopRunRequest(run_id=run_id))
    if response.success:
        typer.secho(f"✅ Run {run_id} successfully stopped.", fg=typer.colors.GREEN)
        return True
    else:
        raise click.ClickException(f"Run {run_id} couldn't be stopped.")
