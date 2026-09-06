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
"""Flower command line interface `log` command."""


from logging import DEBUG, INFO
from typing import Annotated

import click
import httpx
import typer

from flwr.cli.config_migration import migrate, warn_if_federation_config_overrides
from flwr.cli.constant import FEDERATION_CONFIG_HELP_MESSAGE
from flwr.cli.flower_config import read_superlink_connection
from flwr.cli.typing import SuperLinkConnection
from flwr.proto.control_pb2 import StreamLogsRequest  # pylint: disable=E0611
from flwr.supercore import log as logger
from flwr.supercore.control import ControlHttpClient

from .utils import flwr_cli_exc_handler, init_http_client_from_connection

_SHOW_LOGS_TIMEOUT = 5.0


class AllLogsRetrieved(BaseException):
    """Exception raised when all available logs have been retrieved.

    This exception is used internally to signal that the log stream has reached the end
    and all logs have been successfully retrieved.
    """


def start_stream(run_id: int, stub: ControlHttpClient) -> None:
    """Start log streaming for a given run ID.

    Parameters
    ----------
    run_id : int
        The unique identifier of the run to stream logs from.
    stub : ControlHttpClient
        The HTTP client for Control API communication.
    """
    after_timestamp = 0.0
    try:
        logger(INFO, "Starting logstream for run_id `%s`", run_id)
        stream_logs(run_id, stub, after_timestamp)
    except KeyboardInterrupt:
        logger(INFO, "Exiting logstream")
    except AllLogsRetrieved:
        pass


def stream_logs(
    run_id: int,
    stub: ControlHttpClient,
    after_timestamp: float,
) -> float:
    """Stream logs from the beginning of a run.

    Parameters
    ----------
    run_id : int
        The identifier of the run.
    stub : ControlHttpClient
        The HTTP client for Control API communication.
    after_timestamp : float
        The timestamp to start streaming logs from.

    Returns
    -------
    float
        The latest timestamp from the streamed logs or the provided `after_timestamp`
        if no logs are returned.
    """
    req = StreamLogsRequest(run_id=run_id, after_timestamp=after_timestamp)

    latest_timestamp = 0.0
    res = None

    with flwr_cli_exc_handler():
        try:
            for res in stub.StreamLogs(req):
                print(res.log_output, end="")
            raise AllLogsRetrieved()
        finally:
            if res is not None:
                latest_timestamp = res.latest_timestamp

    return max(latest_timestamp, after_timestamp)


def print_logs(run_id: int, stub: ControlHttpClient) -> None:
    """Print logs from the beginning of a run.

    Parameters
    ----------
    run_id : int
        The unique identifier of the run to retrieve logs from.
    stub : ControlHttpClient
        The HTTP client for Control API communication.
    """
    req = StreamLogsRequest(run_id=run_id, after_timestamp=0.0)

    with flwr_cli_exc_handler():
        try:
            for res in stub.StreamLogs(req, read_timeout=_SHOW_LOGS_TIMEOUT):
                print(res.log_output)
                break
        except httpx.ReadTimeout:
            pass


def log(
    ctx: typer.Context,
    run_id: Annotated[
        int,
        typer.Argument(help="The Flower run ID to query"),
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
    stream: Annotated[
        bool,
        typer.Option(
            "--stream/--show",
            help="Flag to stream or print logs from the Flower run",
        ),
    ] = True,
) -> None:
    """Get logs from a run.

    Retrieve and display logs from a Flower run. Logs can be streamed in real-time (with
    --stream) or printed once (with --show).
    """
    # Warn `--federation-config` is ignored
    warn_if_federation_config_overrides(federation_config_overrides)

    # Migrate legacy usage if any
    migrate(superlink, args=ctx.args)

    # Read superlink connection configuration
    superlink_connection = read_superlink_connection(superlink)

    try:
        _log_with_control_api(superlink_connection, run_id, stream)
    except Exception as err:  # pylint: disable=broad-except
        raise click.ClickException(str(err)) from None


def _log_with_control_api(
    superlink_connection: SuperLinkConnection,
    run_id: int,
    stream: bool,
) -> None:
    """Retrieve logs using the Control API.

    Parameters
    ----------
    superlink_connection : SuperLinkConnection
        Superlink connection configuration.
    run_id : int
        The unique identifier of the run to retrieve logs from.
    stream : bool
        If True, stream logs continuously; if False, print once.
    """
    control_client = init_http_client_from_connection(superlink_connection)
    try:
        if stream:
            start_stream(run_id, control_client)
        else:
            logger(INFO, "Printing logstream for run_id `%s`", run_id)
            print_logs(run_id, control_client)
    finally:
        control_client.close()
        logger(DEBUG, "HTTP client closed")
