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
"""Flower command line interface `chat` application."""


import asyncio
import json
from time import monotonic
from typing import cast

import click
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import (
    BufferControl,
    ConditionalContainer,
    Dimension,
    FormattedTextControl,
    HSplit,
    Layout,
    Window,
)
from prompt_toolkit.layout.processors import BeforeInput
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import Frame

from flwr.cli.constant import (
    CHAT_AGENT_INPUT_KEY,
    CHAT_AGENT_NAME,
    CHAT_APP_STYLE,
    CHAT_EXIT_COMMAND,
    CHAT_EXIT_HINT,
    CHAT_EXPERIMENTAL_WARNING,
    CHAT_FAILURE_EVENTS,
    CHAT_FLOWER_AGENT_APP_SPEC,
    CHAT_FLOWER_LOGO,
    CHAT_NEW_COMMAND,
    CHAT_NEW_CONVERSATION_MESSAGE,
    CHAT_SPINNER_FRAMES,
    CHAT_TERMINAL_EVENTS,
    CHAT_TEXT_DELTA_EVENT,
    CHAT_USER_PROMPT,
    CHAT_WELCOME_MESSAGE,
)
from flwr.common.serde import user_config_to_proto
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    StartRunRequest,
    StopRunRequest,
    StreamRunEventsRequest,
)
from flwr.proto.control_pb2_grpc import ControlStub
from flwr.proto.task_pb2 import TaskEvent  # pylint: disable=E0611
from flwr.supercore.typing import JSONObject

from .utils import flwr_cli_grpc_exc_handler


class ChatApplication:  # pylint: disable=too-many-instance-attributes
    """Persistent full-screen Flower Chat application."""

    def __init__(self, stub: ControlStub, federation: str | None) -> None:
        self.stub = stub
        self.federation = federation
        self.series_id: int | None = None
        self.run_id: int | None = None
        self.busy = False
        self.cancel_requested = False
        self.transcript: list[tuple[str, str]] = []
        self.wrapped_transcript: list[tuple[str, str]] = []
        self.status = ""
        self.input_buffer = Buffer()
        self.application = self._create_application()

    def run(self) -> None:
        """Run the application until the user exits."""
        self.application.run()

    def _create_application(self) -> Application[None]:
        """Create the persistent full-screen layout."""
        key_bindings = KeyBindings()
        logo_lines = CHAT_FLOWER_LOGO.splitlines()

        @key_bindings.add("enter")
        def _submit_prompt(event: KeyPressEvent) -> None:
            self._submit_prompt(event)

        @key_bindings.add("c-c")
        def _interrupt_prompt(event: KeyPressEvent) -> None:
            self._interrupt_prompt(event)

        @key_bindings.add("c-d")
        def _ignore_eof(_: KeyPressEvent) -> None:
            pass

        welcome = Frame(
            Window(
                FormattedTextControl(
                    [
                        *[("class:logo", f"{line}\n") for line in logo_lines],
                        ("", "\n"),
                        ("class:notice", CHAT_EXPERIMENTAL_WARNING),
                        ("class:welcome", f"\n{CHAT_WELCOME_MESSAGE}."),
                        ("", f"\n{CHAT_EXIT_HINT}"),
                    ]
                ),
                height=len(logo_lines) + 4,
            ),
            style="class:agent.prompt",
        )
        self.transcript_window = Window(
            FormattedTextControl(
                self._render_transcript,
                get_cursor_position=self._transcript_cursor,
                show_cursor=False,
            ),
            wrap_lines=False,
            always_hide_cursor=True,
        )
        status = Window(
            FormattedTextControl(self._render_status),
            height=1,
        )
        status_gap = ConditionalContainer(
            Window(height=1),
            filter=Condition(lambda: bool(self.status)),
        )
        prompt = Window(
            BufferControl(
                buffer=self.input_buffer,
                input_processors=[
                    BeforeInput(CHAT_USER_PROMPT, style="class:user.prompt")
                ],
            ),
            height=Dimension(min=1),
            dont_extend_height=True,
            wrap_lines=True,
            style="class:prompt.background",
        )
        chat_window = HSplit(
            [self.transcript_window, status, status_gap],
            style="class:content",
        )
        agent_name = Window(
            FormattedTextControl([("class:agent.name", f" ✿ {CHAT_AGENT_NAME} ")]),
            height=1,
            style="class:content",
        )
        agent_separator = Window(
            height=1,
            char="─",
            style="class:agent.separator",
        )
        return Application[None](
            layout=Layout(
                HSplit(
                    [
                        welcome,
                        chat_window,
                        agent_name,
                        agent_separator,
                        prompt,
                    ]
                ),
                focused_element=prompt,
            ),
            key_bindings=key_bindings,
            style=Style.from_dict(CHAT_APP_STYLE),
            full_screen=True,
            mouse_support=True,
            refresh_interval=0.1,
        )

    def _submit_prompt(self, event: KeyPressEvent) -> None:
        """Handle a prompt submitted from the input buffer."""
        if self.busy:
            return

        prompt = self.input_buffer.text
        self.input_buffer.reset()
        stripped_prompt = prompt.strip()
        if not stripped_prompt:
            return
        if stripped_prompt.lower() == CHAT_EXIT_COMMAND:
            event.app.exit()
            return
        if stripped_prompt.lower() == CHAT_NEW_COMMAND:
            self.series_id = None
            self._append_transcript(
                "class:notice",
                f"{CHAT_NEW_CONVERSATION_MESSAGE}\n\n",
            )
            return

        self._append_user_message(prompt)
        self.busy = True
        self.cancel_requested = False
        self.status = "Thinking..."
        event.app.create_background_task(self._run_prompt(prompt))
        event.app.invalidate()

    def _interrupt_prompt(self, event: KeyPressEvent) -> None:
        """Exit while idle or stop the active run."""
        if not self.busy:
            if self.input_buffer.text:
                self.input_buffer.reset()
                event.app.invalidate()
                return
            event.app.exit()
            return

        self.cancel_requested = True
        if self.run_id is not None:
            event.app.create_background_task(
                asyncio.to_thread(self._stop_run, self.run_id)
            )

    async def _run_prompt(self, prompt: str) -> None:
        """Run one blocking chat request outside the UI event loop."""
        try:
            await asyncio.to_thread(self._run_prompt_sync, prompt)
        except click.ClickException as exc:
            self._append_transcript("class:error", f"Error: {exc.format_message()}\n\n")
        finally:
            self.run_id = None
            self.busy = False
            self.cancel_requested = False
            self.status = ""
            self.application.layout.focus(self.input_buffer)
            self.application.invalidate()

    def _run_prompt_sync(self, prompt: str) -> None:
        """Start and stream one Flower AgentApp run."""
        self.run_id, self.series_id = start_chat_run(
            self.stub, prompt, self.federation, self.series_id
        )

        if self.cancel_requested:
            self._stop_run(self.run_id)
            return

        response_started = False
        terminal_event_seen = False
        req_events = StreamRunEventsRequest(run_id=self.run_id)
        with flwr_cli_grpc_exc_handler():
            for res_events in self.stub.StreamRunEvents(req_events):
                event_type, payload = parse_task_event(res_events.task_event)
                if event_type == CHAT_TEXT_DELTA_EVENT:
                    delta = payload.get("delta")
                    if isinstance(delta, str):
                        if not response_started:
                            response_started = True
                            self.status = ""
                        self._append_transcript("", delta)
                elif event_type in CHAT_FAILURE_EVENTS:
                    raise click.ClickException(format_failure_event(payload))
                elif event_type in CHAT_TERMINAL_EVENTS:
                    terminal_event_seen = True

        if response_started:
            self._append_transcript("", "\n\n")
        if not terminal_event_seen and not self.cancel_requested:
            raise click.ClickException(
                "Chat run ended before the agent response completed."
            )

    def _stop_run(self, run_id: int) -> None:
        """Stop the active run and report failures in the transcript."""
        try:
            with flwr_cli_grpc_exc_handler():
                response = self.stub.StopRun(request=StopRunRequest(run_id=run_id))
            if not response.success:
                self._append_transcript(
                    "class:error", f"Warning: run {run_id} could not be stopped.\n\n"
                )
        except click.ClickException as exc:
            self._append_transcript(
                "class:error",
                f"Warning: failed to stop run {run_id}: {exc.format_message()}\n\n",
            )

    def _append_transcript(self, style: str, text: str) -> None:
        """Append text and request a screen redraw."""
        self.transcript.append((style, text))
        self.application.invalidate()

    def _append_user_message(self, prompt: str) -> None:
        """Append a full-width highlighted user message."""
        for line_index, line in enumerate(prompt.split("\n")):
            prefix = (
                CHAT_USER_PROMPT
                if line_index == 0
                else " " * get_cwidth(CHAT_USER_PROMPT)
            )
            self.transcript.append(("class:user.message", f"{prefix}{line}\n"))
        self.transcript.append(("", "\n"))
        self.application.invalidate()

    def _get_terminal_width(self) -> int:
        """Return the current terminal width."""
        return max(1, self.application.output.get_size().columns)

    def _render_status(self) -> list[tuple[str, str]]:
        """Return the animated status line."""
        if not self.status:
            return []
        frame = CHAT_SPINNER_FRAMES[int(monotonic() * 10) % len(CHAT_SPINNER_FRAMES)]
        return [("class:status", f"{frame} {self.status}")]

    def _render_transcript(self) -> list[tuple[str, str]]:
        """Return transcript text wrapped to the current terminal width."""
        self.wrapped_transcript = _wrap_transcript_fragments(
            self.transcript, self._get_terminal_width()
        )
        return self.wrapped_transcript

    def _transcript_cursor(self) -> Point:
        """Keep the transcript scrolled to its last line."""
        wrapped_text = "".join(text for _, text in self.wrapped_transcript)
        lines = wrapped_text.split("\n")
        last_line_index = len(lines) - 1
        render_info = self.transcript_window.render_info
        if render_info is not None:
            bottom_scroll = render_info.content_height - render_info.window_height
            if self.transcript_window.vertical_scroll < bottom_scroll:
                return Point(
                    x=0,
                    y=self.transcript_window.vertical_scroll,
                )

        last_line_width = get_cwidth(lines[-1])
        return Point(x=last_line_width, y=last_line_index)


def parse_task_event(task_event: TaskEvent) -> tuple[str, JSONObject]:
    """Return an event type and object payload."""
    event_type = task_event.event
    try:
        raw_payload = json.loads(task_event.data)
    except json.JSONDecodeError:
        raw_payload = {}
    payload = cast(JSONObject, raw_payload) if isinstance(raw_payload, dict) else {}
    if not event_type:
        event_type = cast(str, payload.get("type", ""))
    return event_type, payload


def start_chat_run(
    stub: ControlStub,
    prompt: str,
    federation: str | None,
    series_id: int | None,
) -> tuple[int, int | None]:
    """Start one Flower AgentApp run."""
    req = StartRunRequest(
        app_spec=CHAT_FLOWER_AGENT_APP_SPEC,
        override_config=user_config_to_proto({CHAT_AGENT_INPUT_KEY: prompt}),
        federation=federation or "",
    )
    if series_id is not None:
        req.series_id = series_id

    with flwr_cli_grpc_exc_handler():
        res = stub.StartRun(req)

    if not res.HasField("run_id"):
        raise click.ClickException("Failed to start chat run.")
    if res.HasField("series_id"):
        series_id = cast(int, res.series_id)
    return cast(int, res.run_id), series_id


def format_failure_event(payload: JSONObject) -> str:
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


def _wrap_transcript_fragments(
    fragments: list[tuple[str, str]], width: int
) -> list[tuple[str, str]]:
    """Wrap formatted transcript fragments to the transcript width."""
    wrapped_fragments: list[tuple[str, str]] = []
    current_width = 0
    for style, text in fragments:
        chunk: list[str] = []
        for char in text:
            if char == "\n":
                if chunk:
                    wrapped_fragments.append((style, "".join(chunk)))
                    chunk = []
                if style == "class:user.message" and current_width < width:
                    wrapped_fragments.append((style, " " * (width - current_width)))
                wrapped_fragments.append((style, char))
                current_width = 0
                continue

            char_width = get_cwidth(char)
            if current_width and current_width + char_width > width:
                if chunk:
                    wrapped_fragments.append((style, "".join(chunk)))
                    chunk = []
                if style == "class:user.message" and current_width < width:
                    wrapped_fragments.append((style, " " * (width - current_width)))
                wrapped_fragments.append(("", "\n"))
                current_width = 0
            chunk.append(char)
            current_width += char_width
        if chunk:
            wrapped_fragments.append((style, "".join(chunk)))
    return wrapped_fragments
