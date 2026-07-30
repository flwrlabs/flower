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
"""Flower command line interface `chat` command."""


import sys

import click
import typer

from flwr.cli.constant import (
    CHAT_EXIT_COMMAND,
    CHAT_EXIT_HINT,
    CHAT_FAILURE_EVENTS,
    CHAT_NEW_COMMAND,
    CHAT_NEW_CONVERSATION_MESSAGE,
    CHAT_SUPERGRID_CONNECTION_NAME,
    CHAT_TERMINAL_EVENTS,
    CHAT_TEXT_DELTA_EVENT,
    CHAT_USER_PROMPT,
    CHAT_WELCOME_MESSAGE,
)
from flwr.cli.flower_config import read_superlink_connection
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListFederationsRequest,
    StopRunRequest,
    StreamRunEventsRequest,
)
from flwr.proto.control_pb2_grpc import ControlStub

from .chat_app import (
    ChatApplication,
    format_failure_event,
    parse_task_event,
    start_chat_run,
)
from .utils import flwr_cli_grpc_exc_handler, init_channel_from_connection


def chat() -> None:
    """Start an interactive chat session with the Flower agent."""
    superlink_connection = read_superlink_connection(CHAT_SUPERGRID_CONNECTION_NAME)

    channel = init_channel_from_connection(superlink_connection)
    stub = ControlStub(channel)
    try:
        # Verify stored credentials before showing the interactive prompt.
        with flwr_cli_grpc_exc_handler():
            stub.ListFederations(ListFederationsRequest())
        if sys.stdin.isatty() and sys.stdout.isatty():
            ChatApplication(stub, superlink_connection.federation).run()
        else:
            typer.echo(CHAT_WELCOME_MESSAGE)
            typer.echo(f"Flower Chat. {CHAT_EXIT_HINT}")
            _run_interactive_shell(stub, superlink_connection.federation)
    finally:
        channel.close()


def _run_interactive_shell(stub: ControlStub, federation: str | None) -> None:
    """Run the non-TTY prompt-response loop."""
    series_id: int | None = None
    while True:
        try:
            prompt = input(CHAT_USER_PROMPT)
        except EOFError:
            typer.echo()
            if not sys.stdin.isatty():
                return
            continue
        except KeyboardInterrupt:
            typer.echo()
            return

        stripped_prompt = prompt.strip()
        if not stripped_prompt:
            continue
        if stripped_prompt.lower() == CHAT_EXIT_COMMAND:
            return
        if stripped_prompt.lower() == CHAT_NEW_COMMAND:
            series_id = None
            typer.echo(CHAT_NEW_CONVERSATION_MESSAGE)
            continue

        run_id: int | None = None
        try:
            run_id, series_id = start_chat_run(stub, prompt, federation, series_id)
            _stream_agent_response(stub, run_id)
        except KeyboardInterrupt:
            typer.echo()
            if run_id is not None:
                try:
                    with flwr_cli_grpc_exc_handler():
                        response = stub.StopRun(request=StopRunRequest(run_id=run_id))
                    if not response.success:
                        typer.echo(
                            f"Warning: run {run_id} could not be stopped.",
                            err=True,
                        )
                except click.ClickException as exc:
                    typer.echo(
                        f"Warning: failed to stop run {run_id}: {exc.format_message()}",
                        err=True,
                    )
            continue


def _stream_agent_response(stub: ControlStub, run_id: int) -> None:
    """Stream one AgentApp response to stdout."""
    terminal_event_seen = False
    response_started = False
    try:
        req = StreamRunEventsRequest(run_id=run_id)
        with flwr_cli_grpc_exc_handler():
            for res in stub.StreamRunEvents(req):
                event_type, payload = parse_task_event(res.task_event)

                if event_type == CHAT_TEXT_DELTA_EVENT:
                    delta = payload.get("delta")
                    if isinstance(delta, str):
                        if not response_started:
                            response_started = True
                        typer.echo(delta, nl=False)
                elif event_type in CHAT_FAILURE_EVENTS:
                    raise click.ClickException(format_failure_event(payload))
                elif event_type in CHAT_TERMINAL_EVENTS:
                    terminal_event_seen = True
    finally:
        if response_started:
            typer.echo()

    if not terminal_event_seen:
        raise click.ClickException(
            "Chat run ended before the agent response completed."
        )
