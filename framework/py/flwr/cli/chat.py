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
from typing import cast

import click
import typer
from rich.console import Console
from rich.status import Status

from flwr.cli.flower_config import read_superlink_connection
from flwr.cli.typing import SuperLinkConnection
from flwr.common.constant import AuthnType
from flwr.common.serde import user_config_to_proto
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListFederationsRequest,
    StartRunRequest,
    StreamRunEventsRequest,
)
from flwr.proto.control_pb2_grpc import ControlStub
from flwr.supercore.typing import JSONObject

from .auth_plugin import CliAuthPlugin
from .utils import (
    flwr_cli_grpc_exc_handler,
    get_authn_type,
    init_channel_from_connection,
    load_cli_auth_plugin_from_connection,
)

_BUILTIN_AGENT_APP_SPEC = "@flwrlabs/flwr-agent"
_SUPERGRID_CONNECTION_NAME = "supergrid"
_AGENT_INPUT_KEY = "agent.input"
_EXIT_COMMANDS = {"/quit"}
_TEXT_DELTA_EVENT = "response.output_text.delta"
_TERMINAL_EVENTS = {"response.completed", "response.incomplete"}
_FAILURE_EVENTS = {"error", "response.failed"}
_AGENT_COLOR_HEX = "#f2b607"
_USER_PROMPT = "You> "
_AGENT_COLOR = "\033[38;2;242;182;7m"
_ANSI_RESET = "\033[0m"
_AGENT_PROMPT = f"{_AGENT_COLOR}Agent> "


def chat() -> None:
    """Start an interactive chat session with the Flower agent."""
    superlink_connection = read_superlink_connection(_SUPERGRID_CONNECTION_NAME)
    auth_plugin = _load_logged_in_auth_plugin(superlink_connection)

    channel = init_channel_from_connection(superlink_connection, auth_plugin)
    stub = ControlStub(channel)
    try:
        _verify_authenticated(stub)
        typer.secho(
            "Flower Chat. Type /quit to leave.",
            fg=typer.colors.BLUE,
        )
        _run_interactive_shell(stub, superlink_connection.federation)
    finally:
        channel.close()


def _load_logged_in_auth_plugin(
    superlink_connection: SuperLinkConnection,
) -> CliAuthPlugin:
    """Load a logged-in auth plugin or fail before the chat prompt starts."""
    address = superlink_connection.address
    if address is None:
        raise click.ClickException("Please run `flwr login supergrid` first.")

    authn_type = get_authn_type(address)
    if authn_type == AuthnType.NOOP:
        raise click.ClickException("Please run `flwr login supergrid` first.")

    auth_plugin = load_cli_auth_plugin_from_connection(address, authn_type)
    auth_plugin.load_tokens()
    try:
        auth_plugin.write_tokens_to_metadata([])
    except click.ClickException as exc:
        raise click.ClickException("Please run `flwr login supergrid` first.") from exc

    return auth_plugin


def _verify_authenticated(stub: ControlStub) -> None:
    """Verify the stored credentials against the SuperGrid before prompting."""
    with flwr_cli_grpc_exc_handler():
        stub.ListFederations(ListFederationsRequest())


def _run_interactive_shell(stub: ControlStub, federation: str | None) -> None:
    """Run the prompt-response loop."""
    while True:
        try:
            prompt = input(_USER_PROMPT)
        except (EOFError, KeyboardInterrupt):
            typer.echo()
            return

        stripped_prompt = prompt.strip()
        if not stripped_prompt:
            continue
        if stripped_prompt.lower() in _EXIT_COMMANDS:
            return

        _run_prompt(stub, prompt, federation)


def _run_prompt(stub: ControlStub, prompt: str, federation: str | None) -> None:
    """Submit one prompt and stream the response."""
    status = Console().status(
        "Thinking...", spinner="dots", spinner_style=_AGENT_COLOR_HEX
    )
    status.start()
    try:
        run_id = _start_agent_run(stub, prompt, federation)
        _stream_agent_response(stub, run_id, status)
    finally:
        status.stop()


def _start_agent_run(
    stub: ControlStub,
    prompt: str,
    federation: str | None,
) -> int:
    """Start one built-in AgentApp run for the given prompt."""
    req = StartRunRequest(
        app_spec=_BUILTIN_AGENT_APP_SPEC,
        override_config=user_config_to_proto({_AGENT_INPUT_KEY: prompt}),
        federation=federation or "",
    )
    with flwr_cli_grpc_exc_handler():
        res = stub.StartRun(req)

    if not res.HasField("run_id"):
        raise click.ClickException("Failed to start chat run.")
    return cast(int, res.run_id)


def _stream_agent_response(stub: ControlStub, run_id: int, status: Status) -> None:
    """Stream one AgentApp response to stdout."""
    terminal_event_seen = False
    word_stream = _WordStreamPrinter(status)
    try:
        req = StreamRunEventsRequest(run_id=run_id)
        with flwr_cli_grpc_exc_handler():
            for res in stub.StreamRunEvents(req):
                event_type = res.task_event.event
                payload = _load_task_event_data(res.task_event.data)
                if not event_type:
                    event_type = cast(str, payload.get("type", ""))

                if event_type == _TEXT_DELTA_EVENT:
                    delta = payload.get("delta")
                    if isinstance(delta, str):
                        word_stream.write_delta(delta)
                elif event_type in _FAILURE_EVENTS:
                    raise click.ClickException(_format_failure_event(payload))
                elif event_type in _TERMINAL_EVENTS:
                    terminal_event_seen = True
                    break
    finally:
        word_stream.finish()

    if not terminal_event_seen:
        raise click.ClickException(
            "Chat run ended before the agent response completed."
        )


def _load_task_event_data(data: str) -> JSONObject:
    """Parse a task event JSON payload."""
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return cast(JSONObject, payload)


class _WordStreamPrinter:
    """Print streamed text at word boundaries."""

    def __init__(self, status: Status) -> None:
        self._status = status
        self._buffer = ""
        self.response_started = False

    def write_delta(self, delta: str) -> None:
        """Print complete words from one text delta."""
        if not self.response_started:
            self._status.stop()
            print(_AGENT_PROMPT, end="", flush=True)
            self.response_started = True

        self._buffer += delta
        boundary = _last_whitespace_boundary(self._buffer)
        if boundary is None:
            return

        print(self._buffer[:boundary], end="", flush=True)
        self._buffer = self._buffer[boundary:]

    def finish(self) -> None:
        """Flush any pending text and finish the response line."""
        if not self.response_started:
            return
        if self._buffer:
            print(self._buffer, end="", flush=True)
            self._buffer = ""
        print(_ANSI_RESET)


def _last_whitespace_boundary(text: str) -> int | None:
    """Return the index after the last whitespace character, if present."""
    for idx in range(len(text) - 1, -1, -1):
        if text[idx].isspace():
            return idx + 1
    return None


def _format_failure_event(payload: JSONObject) -> str:
    """Return a concise failure message from a streamed event payload."""
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message

    message = payload.get("message")
    if isinstance(message, str) and message:
        return message

    return "Agent response failed."
