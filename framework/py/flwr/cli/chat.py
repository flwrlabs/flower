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

from flwr.cli.constant import (
    CHAT_AGENT_COLOR_HEX,
    CHAT_AGENT_INPUT_KEY,
    CHAT_AGENT_PROMPT,
    CHAT_ANSI_RESET,
    CHAT_EXIT_COMMAND,
    CHAT_FAILURE_EVENTS,
    CHAT_FLOWER_AGENT_APP_SPEC,
    CHAT_LOGIN_REQUIRED_MESSAGE,
    CHAT_SUPERGRID_CONNECTION_NAME,
    CHAT_TERMINAL_EVENTS,
    CHAT_TEXT_DELTA_EVENT,
    CHAT_USER_PROMPT,
)
from flwr.cli.flower_config import read_superlink_connection
from flwr.common.constant import AuthnType
from flwr.common.serde import user_config_to_proto
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListFederationsRequest,
    StartRunRequest,
    StopRunRequest,
    StreamRunEventsRequest,
)
from flwr.proto.control_pb2_grpc import ControlStub
from flwr.supercore.typing import JSONObject

from .utils import (
    flwr_cli_grpc_exc_handler,
    get_authn_type,
    init_channel_from_connection,
    load_cli_auth_plugin_from_connection,
)


def chat() -> None:
    """Start an interactive chat session with the Flower agent."""
    superlink_connection = read_superlink_connection(CHAT_SUPERGRID_CONNECTION_NAME)

    # Reject insecure connections before loading stored auth tokens.
    if superlink_connection.insecure:
        raise click.ClickException(
            "`flwr chat` requires TLS to be enabled. `insecure` must NOT be set to "
            "`true` in the federation configuration."
        )

    # Load a logged-in auth plugin or fail before the chat prompt starts.
    address = superlink_connection.address
    if address is None:
        raise click.ClickException(CHAT_LOGIN_REQUIRED_MESSAGE)

    authn_type = get_authn_type(address)
    if authn_type == AuthnType.NOOP:
        raise click.ClickException(CHAT_LOGIN_REQUIRED_MESSAGE)

    auth_plugin = load_cli_auth_plugin_from_connection(address, authn_type)
    auth_plugin.load_tokens()
    try:
        auth_plugin.write_tokens_to_metadata([])
    except click.ClickException as exc:
        raise click.ClickException(CHAT_LOGIN_REQUIRED_MESSAGE) from exc

    channel = init_channel_from_connection(superlink_connection, auth_plugin)
    stub = ControlStub(channel)
    try:
        # Verify stored credentials before showing the interactive prompt.
        with flwr_cli_grpc_exc_handler():
            stub.ListFederations(ListFederationsRequest())
        typer.secho(
            f"Flower Chat. Type {CHAT_EXIT_COMMAND} to leave.",
            fg=typer.colors.BLUE,
        )
        _run_interactive_shell(stub, superlink_connection.federation)
    finally:
        channel.close()


def _run_interactive_shell(stub: ControlStub, federation: str | None) -> None:
    """Run the prompt-response loop."""
    while True:
        try:
            prompt = input(CHAT_USER_PROMPT)
        except (EOFError, KeyboardInterrupt):
            typer.echo()
            return
        except KeyboardInterrupt:
            typer.echo()
            continue

        stripped_prompt = prompt.strip()
        if not stripped_prompt:
            continue
        if stripped_prompt.lower() == CHAT_EXIT_COMMAND:
            return

        with Console().status(
            "Thinking...", spinner="dots", spinner_style=CHAT_AGENT_COLOR_HEX
        ) as status:
            # Start one Flower AgentApp run for the submitted prompt.
            req = StartRunRequest(
                app_spec=CHAT_FLOWER_AGENT_APP_SPEC,
                override_config=user_config_to_proto({CHAT_AGENT_INPUT_KEY: prompt}),
                federation=federation or "",
            )
            with flwr_cli_grpc_exc_handler():
                res = stub.StartRun(req)

            if not res.HasField("run_id"):
                raise click.ClickException("Failed to start chat run.")
            _stream_agent_response(stub, cast(int, res.run_id), status)


def _stop_agent_run(stub: ControlStub, run_id: int) -> None:
    """Stop one interrupted AgentApp run."""
    try:
        with flwr_cli_grpc_exc_handler():
            stub.StopRun(StopRunRequest(run_id=run_id))
    except click.ClickException:
        pass


def _stream_agent_response(stub: ControlStub, run_id: int, status: Status) -> None:
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
                            print(CHAT_AGENT_PROMPT, end="", flush=True)
                            response_started = True
                        print(delta, end="", flush=True)
                elif event_type in CHAT_FAILURE_EVENTS:
                    raise click.ClickException(_format_failure_event(payload))
                elif event_type in CHAT_TERMINAL_EVENTS:
                    terminal_event_seen = True
                    break
    finally:
        if response_started:
            print(CHAT_ANSI_RESET)

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
