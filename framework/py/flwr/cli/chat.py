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


import json
import sys
from typing import cast

import click
import typer
from rich.console import Console
from rich.status import Status
from rich.theme import Theme

from flwr.cli.constant import (
    CHAT_AGENT_INPUT_KEY,
    CHAT_EXIT_COMMAND,
    CHAT_FAILURE_EVENTS,
    CHAT_FLOWER_AGENT_APP_SPEC,
    CHAT_SUPERGRID_CONNECTION_NAME,
    CHAT_TERMINAL_EVENTS,
    CHAT_TEXT_DELTA_EVENT,
    CHAT_USER_PROMPT,
)
from flwr.cli.flower_config import read_superlink_connection
from flwr.common.serde import user_config_to_proto
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListFederationsRequest,
    StartRunRequest,
    StopRunRequest,
    StreamRunEventsRequest,
)
from flwr.proto.control_pb2_grpc import ControlStub
from flwr.supercore.typing import JSONObject

from .utils import flwr_cli_grpc_exc_handler, init_channel_from_connection

# Use ANSI color names so terminals can map them to their active palette.
_CHAT_THEME = Theme(
    {
        "user.prompt": "bold blue",
        "agent.prompt": "bold magenta",
        "agent.text": "default",
        "status": "magenta",
        "notice": "bold",
        "error": "bold bright_red",
    }
)


def chat() -> None:
    """Start an interactive chat session with the Flower agent."""
    console = Console(theme=_CHAT_THEME)
    console.print(
        "Note: `flwr chat` is experimental and subject to change.",
        style="notice",
    )
    superlink_connection = read_superlink_connection(CHAT_SUPERGRID_CONNECTION_NAME)

    channel = init_channel_from_connection(superlink_connection)
    stub = ControlStub(channel)
    try:
        # Verify stored credentials before showing the interactive prompt.
        with flwr_cli_grpc_exc_handler():
            stub.ListFederations(ListFederationsRequest())
        console.print(
            f"Flower Chat. Type {CHAT_EXIT_COMMAND} or press Ctrl-C to leave.",
            style="agent.prompt",
        )
        _run_interactive_shell(stub, superlink_connection.federation, console)
    finally:
        channel.close()


def _run_interactive_shell(
    stub: ControlStub, federation: str | None, console: Console
) -> None:
    """Run the prompt-response loop."""
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

        run_id: int | None = None
        try:
            with console.status(
                "Thinking...", spinner="dots", spinner_style="status"
            ) as status:
                # Start one Flower AgentApp run for the submitted prompt.
                req = StartRunRequest(
                    app_spec=CHAT_FLOWER_AGENT_APP_SPEC,
                    override_config=user_config_to_proto(
                        {CHAT_AGENT_INPUT_KEY: prompt}
                    ),
                    federation=federation or "",
                )
                with flwr_cli_grpc_exc_handler():
                    res = stub.StartRun(req)

                if not res.HasField("run_id"):
                    raise click.ClickException("Failed to start chat run.")
                run_id = cast(int, res.run_id)
                _stream_agent_response(stub, run_id, status, console)
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
                        f"Warning: failed to stop run {run_id}: "
                        f"{exc.format_message()}",
                        err=True,
                    )
            continue


def _stream_agent_response(
    stub: ControlStub, run_id: int, status: Status, console: Console
) -> None:
    """Stream one AgentApp response to stdout."""
    terminal_event_seen = False
    response_started = False
    try:
        req = StreamRunEventsRequest(run_id=run_id)
        with flwr_cli_grpc_exc_handler():
            for res in stub.StreamRunEvents(req):
                event_type = res.task_event.event

                # Parse event payloads defensively; event names can carry the type.
                try:
                    raw_payload = json.loads(res.task_event.data)
                except json.JSONDecodeError:
                    raw_payload = {}
                payload = (
                    cast(JSONObject, raw_payload)
                    if isinstance(raw_payload, dict)
                    else {}
                )
                if not event_type:
                    event_type = cast(str, payload.get("type", ""))

                # Print streamed text deltas as the agent response.
                if event_type == CHAT_TEXT_DELTA_EVENT:
                    delta = payload.get("delta")
                    if isinstance(delta, str):
                        if not response_started:
                            status.stop()
                            console.print("Agent> ", style="agent.prompt", end="")
                            response_started = True
                        console.print(
                            delta,
                            style="agent.text",
                            end="",
                            markup=False,
                            highlight=False,
                        )
                elif event_type in CHAT_FAILURE_EVENTS:
                    raise click.ClickException(_format_failure_event(payload))
                elif event_type in CHAT_TERMINAL_EVENTS:
                    terminal_event_seen = True
                    break
    finally:
        if response_started:
            console.print()

    if not terminal_event_seen:
        raise click.ClickException(
            "Chat run ended before the agent response completed."
        )


def _format_failure_event(payload: JSONObject) -> str:
    """Return a concise failure message from a streamed event payload."""
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message

    response = payload.get("response")
    if isinstance(response, dict):
        error = response.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                return message

    message = payload.get("message")
    if isinstance(message, str) and message:
        return message

    return "Agent response failed."
