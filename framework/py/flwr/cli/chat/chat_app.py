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

# pylint: disable=too-many-lines

import asyncio
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from threading import Lock
from time import monotonic
from types import NoneType
from typing import Any, cast

import click
import requests
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer, CompletionState
from prompt_toolkit.completion import (
    CompleteEvent,
    Completer,
    Completion,
    ThreadedCompleter,
)
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, has_completions, is_done
from prompt_toolkit.formatted_text import StyleAndTextTuples
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
from prompt_toolkit.layout.menus import CompletionsMenuControl
from prompt_toolkit.layout.mouse_handlers import MouseHandler
from prompt_toolkit.layout.processors import BeforeInput
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import Frame

from flwr.cli.constant import (
    CHAT_AGENT_INPUT_KEY,
    CHAT_AGENT_NAME,
    CHAT_AGENTS_API_PATH,
    CHAT_APP_STYLE,
    CHAT_COMMANDS,
    CHAT_DEFAULT_FEDERATION_NAME,
    CHAT_EXIT_COMMAND,
    CHAT_EXIT_HINT,
    CHAT_EXPERIMENTAL_WARNING,
    CHAT_FAILURE_EVENTS,
    CHAT_FEDERATION_COMMAND,
    CHAT_FLOWER_LOGO,
    CHAT_HELP_COMMAND,
    CHAT_HISTORY_COMMAND,
    CHAT_NEW_COMMAND,
    CHAT_REASONING_DELTA_EVENT,
    CHAT_SPINNER_FRAMES,
    CHAT_TERMINAL_EVENTS,
    CHAT_TEXT_DELTA_EVENT,
    CHAT_TOOL_CALL_COMPLETED_EVENT,
    CHAT_TOOL_CALL_STARTED_EVENT,
    CHAT_USER_PROMPT,
    CHAT_WEB_SEARCH_CONNECTOR_REF,
    CHAT_WELCOME_MESSAGE,
)
from flwr.common.serde import user_config_to_proto
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    StartRunRequest,
    StopRunRequest,
    StreamRunEventsRequest,
)
from flwr.proto.control_pb2_grpc import ControlStub
from flwr.proto.fab_pb2 import Fab  # pylint: disable=E0611
from flwr.proto.federation_pb2 import Federation  # pylint: disable=E0611
from flwr.proto.task_pb2 import TaskEvent  # pylint: disable=E0611
from flwr.supercore.constant import (
    APP_ID_PATTERN,
    FLOWER_AGENT_APP_ID,
    FLWR_SUPERGRID_API_URL,
)
from flwr.supercore.typing import JSONObject

from ..auth_plugin import CliAuthPlugin, OidcCliPlugin
from ..utils import flwr_cli_grpc_exc_handler
from .chat_federation import complete_federations, select_federation
from .chat_history import HistoryBlock, load_conversation, load_history, render_history
from .chat_transcript import MarkdownBlock, render_markdown


@dataclass
class _DetailsBlock:
    """Collapsible details shown inside the chat transcript."""

    title: str
    body: str = ""
    expanded: bool = False


@dataclass(frozen=True)
class _Agent:
    """Agent available to Flower Chat."""

    app_spec: str
    display_name: str
    description: str
    fab_hash: str | None


def _finalize_markdown_block(block: MarkdownBlock) -> bool:
    """Finalize a Markdown block and report whether it changed."""
    if block.finalized:
        return False
    block.finalized = True
    return True


class _ChatCompleter(Completer):
    """Complete slash commands and agents in the prompt."""

    def __init__(
        self,
        auth_plugin: CliAuthPlugin,
        federation: str,
        federations: list[Federation],
    ) -> None:
        self.auth_plugin = auth_plugin
        self.federation = federation
        self.federations = federations
        self.agents: list[_Agent] | None = None
        self._agents_lock = Lock()

    def load_agents(self) -> list[_Agent]:
        """Load and cache the available agents."""
        with self._agents_lock:
            if self.agents is None:
                self.agents = fetch_chat_agents(self.auth_plugin, self.federation)
            return self.agents

    def set_federation(self, federation: str) -> None:
        """Select a federation and clear cached agent completions."""
        with self._agents_lock:
            self.federation = federation
            self.agents = None

    def get_completions(  # pylint: disable=too-many-return-statements,too-many-branches
        self, document: Document, _complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        """Yield matching commands or agents."""
        text = document.text_before_cursor
        if document.text_after_cursor:
            return

        federation_prefix = f"{CHAT_FEDERATION_COMMAND} "
        if text.lower().startswith(federation_prefix):
            query = text[len(federation_prefix) :]
            if any(char.isspace() for char in query):
                return
            yield from complete_federations(query, self.federations)
            return

        if any(char.isspace() for char in text):
            return
        if text.startswith("/"):
            command_width = max(len(command) for command in CHAT_COMMANDS)
            for command, description in CHAT_COMMANDS.items():
                if command.startswith(text):
                    yield Completion(
                        command,
                        start_position=-len(text),
                        display=f"{command:<{command_width}}        {description}",
                        selected_style="#ffffff bg:#dc8400 noreverse",
                    )
            return

        if not text.startswith("@"):
            return
        try:
            agents = self.load_agents()
        except click.ClickException:
            return

        matches = [
            agent for agent in agents if agent.app_spec.lower().startswith(text.lower())
        ]
        if not matches:
            return
        app_spec_width = max(len(agent.app_spec) for agent in matches)
        for agent in matches:
            yield Completion(
                f"{agent.app_spec} ",
                start_position=-len(text),
                display=(
                    f"{agent.app_spec:<{app_spec_width}}        {agent.description}"
                ),
                selected_style="#ffffff bg:#dc8400 noreverse",
            )


class _FullWidthCompletionsMenuControl(CompletionsMenuControl):
    """Render completion menu rows across the available width."""

    def _get_menu_width(self, max_width: int, _complete_state: CompletionState) -> int:
        """Use all available columns for each completion row."""
        return max_width


class ChatApplication:  # pylint: disable=too-many-instance-attributes
    """Persistent full-screen Flower Chat application."""

    def __init__(
        self,
        stub: ControlStub,
        federations: list[Federation],
        auth_plugin: CliAuthPlugin,
    ) -> None:
        self.stub = stub
        self.federation = next(
            federation.name
            for federation in federations
            if federation.name.endswith(f"/{CHAT_DEFAULT_FEDERATION_NAME}")
        )
        self.federations = federations
        self.series_id: int | None = None
        self.run_id: int | None = None
        self.busy = False
        self.cancel_requested = False
        self.transcript: list[
            tuple[str, str] | _DetailsBlock | HistoryBlock | MarkdownBlock
        ] = []
        self.history_block: HistoryBlock | None = None
        self.history_loading = False
        self.wrapped_transcript: StyleAndTextTuples = []
        self.wrapped_transcript_key: tuple[int, int] | None = None
        self.transcript_revision = 0
        self.follow_transcript = True
        self.status = ""
        self.agent_app_spec = FLOWER_AGENT_APP_ID
        self.agent_fab_hash: str | None = None
        self.agent_name = CHAT_AGENT_NAME
        self.completer = _ChatCompleter(auth_plugin, self.federation, federations)
        self.input_buffer = Buffer(
            completer=ThreadedCompleter(self.completer),
            complete_while_typing=True,
        )
        self.application = self._create_application()

    def run(self) -> None:
        """Run the application until the user exits."""
        self.application.run()

    def _create_application(  # pylint: disable=too-many-locals
        self,
    ) -> Application[None]:
        """Create the persistent full-screen layout."""
        key_bindings = KeyBindings()
        logo_lines = CHAT_FLOWER_LOGO.splitlines()

        # Register prompt submission and interruption shortcuts.
        @key_bindings.add("enter")
        def _submit_prompt(event: KeyPressEvent) -> None:
            if self.history_block is not None:
                if self.history_loading:
                    return
                self.history_loading = True
                event.app.create_background_task(self._confirm_history_selection())
                return
            self._submit_prompt(event)

        @key_bindings.add(
            "up", filter=Condition(lambda: self.history_block is not None)
        )
        def _select_previous_history(_: KeyPressEvent) -> None:
            self._move_history_selection(-1)

        @key_bindings.add(
            "down", filter=Condition(lambda: self.history_block is not None)
        )
        def _select_next_history(_: KeyPressEvent) -> None:
            self._move_history_selection(1)

        @key_bindings.add(
            "escape", filter=Condition(lambda: self.history_block is not None)
        )
        def _cancel_history(_: KeyPressEvent) -> None:
            self._close_history_selection()

        @key_bindings.add("c-c")
        def _interrupt_prompt(event: KeyPressEvent) -> None:
            self._interrupt_prompt(event)

        @key_bindings.add("c-d")
        def _ignore_eof(_: KeyPressEvent) -> None:
            pass

        # Build the fixed welcome header.
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
        # Build the transcript and response status area.
        self.transcript_window = Window(
            FormattedTextControl(
                self._render_transcript,
                get_cursor_position=self._transcript_cursor,
                show_cursor=False,
            ),
            # Wrap manually so scroll offsets map to visual transcript lines.
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
        # Build the user input area.
        prompt = Window(
            BufferControl(
                buffer=self.input_buffer,
                input_processors=[
                    BeforeInput(CHAT_USER_PROMPT, style="class:user.prompt")
                ],
            ),
            # Grow with the draft until the layout constrains and scrolls it.
            height=Dimension(min=1),
            dont_extend_height=True,
            wrap_lines=True,
            style="class:prompt.background",
        )
        completion_menu = ConditionalContainer(
            Window(
                content=_FullWidthCompletionsMenuControl(),
                height=Dimension(min=1, max=4),
                dont_extend_height=True,
                style="class:completion-menu",
            ),
            # show when completions exist AND the application is NOT finished
            filter=Condition(lambda: has_completions() and not is_done()),
        )
        # Combine transcript and status in the main chat area.
        chat_window = HSplit(
            [self.transcript_window, status, status_gap],
            style="class:content",
        )
        # Build the agent label above the input area.
        agent_name = Window(
            FormattedTextControl(self._render_agent_name),
            height=1,
            style="class:content",
        )
        agent_separator = Window(
            height=1,
            char="─",
            style="class:agent.separator",
        )
        # Assemble the full-screen chat layout.
        return Application[None](
            layout=Layout(
                HSplit(
                    [
                        welcome,
                        chat_window,
                        agent_name,
                        agent_separator,
                        prompt,
                        completion_menu,
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
        if self.busy or self.history_loading:
            return

        prompt = self.input_buffer.text
        self.input_buffer.reset()
        stripped_prompt = prompt.strip()
        if not stripped_prompt:
            return
        if self._handle_command(event, stripped_prompt):
            return

        selected_agent, prompt = _extract_agent_selection(prompt)
        if selected_agent is not None:
            try:
                agents = self.completer.load_agents()
            except click.ClickException as exc:
                self._append_transcript(
                    "class:error", f"Error: {exc.format_message()}\n\n"
                )
                event.app.invalidate()
                return
            agent = _find_agent(selected_agent, agents)
            selected_fab_hash = agent.fab_hash if agent is not None else None
            if (
                selected_agent != self.agent_app_spec
                or selected_fab_hash != self.agent_fab_hash
            ):
                self.series_id = None
            self.agent_app_spec = selected_agent
            self.agent_fab_hash = selected_fab_hash
            self.agent_name = (
                agent.display_name if agent is not None else selected_agent
            )
        if not prompt.strip():
            return

        # Start the agent run without blocking the UI event loop.
        self._append_user_message(prompt)
        self.busy = True
        self.cancel_requested = False
        self.status = "Thinking..."
        event.app.create_background_task(
            self._run_prompt(prompt, self.agent_app_spec, self.agent_fab_hash)
        )
        event.app.invalidate()

    def _handle_command(self, event: KeyPressEvent, prompt: str) -> bool:
        """Handle a slash command and return whether the prompt was consumed."""
        command = prompt.lower()
        if command == CHAT_HELP_COMMAND:
            self._append_transcript("class:notice", format_chat_help())
            return True
        if command == CHAT_EXIT_COMMAND:
            event.app.exit()
            return True
        if command == CHAT_NEW_COMMAND:
            self.series_id = None
            self._clear_transcript()
            return True
        if command == CHAT_HISTORY_COMMAND:
            self.history_loading = True
            event.app.create_background_task(self._show_history())
            return True
        if command == CHAT_FEDERATION_COMMAND or command.startswith(
            f"{CHAT_FEDERATION_COMMAND} "
        ):
            return self._handle_federation_command(event, prompt)
        return False

    def _handle_federation_command(self, event: KeyPressEvent, prompt: str) -> bool:
        """Show the federation selector or apply its selection."""
        if prompt.lower() == CHAT_FEDERATION_COMMAND:
            self.input_buffer.text = f"{CHAT_FEDERATION_COMMAND} "
            self.input_buffer.cursor_position = len(self.input_buffer.text)
            self.input_buffer.start_completion(select_first=False)
            event.app.invalidate()
            return True

        try:
            federation_name = select_federation(prompt, self.federations)
        except click.ClickException as exc:
            self._append_transcript("class:error", f"{exc.format_message()}\n\n")
            return True

        if federation_name == self.federation:
            return True

        self.federation = federation_name
        self.completer.set_federation(federation_name)
        self.agent_app_spec = FLOWER_AGENT_APP_ID
        self.agent_fab_hash = None
        self.agent_name = CHAT_AGENT_NAME
        self.series_id = None
        self._clear_transcript()
        return True

    async def _show_history(self) -> None:
        """Show conversation history for the active federation."""
        try:
            block = await asyncio.to_thread(load_history, self.stub, self.federation)
        except click.ClickException as exc:
            self._append_transcript("class:error", f"Error: {exc.format_message()}\n\n")
            return
        finally:
            self.history_loading = False
        if block is None:
            self._append_transcript(
                "class:notice",
                f"No conversation history found for {self.federation}.\n\n",
            )
            return
        self.history_block = block
        self.follow_transcript = True
        self.transcript.append(self.history_block)
        self.transcript_revision += 1
        self.application.invalidate()

    def _move_history_selection(self, offset: int) -> None:
        """Move the highlighted conversation history row."""
        if self.history_block is None:
            return
        self.history_block.selected_index = (
            self.history_block.selected_index + offset
        ) % len(self.history_block.entries)
        self.transcript_revision += 1
        self.application.invalidate()

    async def _confirm_history_selection(self) -> None:
        """Restore the conversation without changing the selected agent."""
        if self.history_block is None:
            self.history_loading = False
            return
        entry = self.history_block.entries[self.history_block.selected_index]
        try:
            messages = await asyncio.to_thread(
                load_conversation, self.stub, entry, self.federation
            )
        except click.ClickException as exc:
            self._close_history_selection()
            self._append_transcript("class:error", f"Error: {exc.format_message()}\n\n")
            return
        finally:
            self.history_loading = False
        self._close_history_selection()
        self._clear_transcript()
        self.series_id = entry.series_id
        for role, text in messages:
            if role == "user":
                self._append_user_message(text)
            else:
                self._append_transcript("", f"{text}\n\n")

    def _close_history_selection(self) -> None:
        """Remove the active conversation history block."""
        if self.history_block is None:
            return
        self.transcript.remove(self.history_block)
        self.history_block = None
        self.follow_transcript = True
        self.transcript_revision += 1
        self.application.invalidate()

    def _interrupt_prompt(self, event: KeyPressEvent) -> None:
        """Exit while idle or stop the active run."""
        if self.history_block is not None:
            self._close_history_selection()
            return
        # Clear a draft or exit when no run is active.
        if not self.busy:
            if self.input_buffer.text:
                self.input_buffer.reset()
                event.app.invalidate()
                return
            event.app.exit()
            return

        # Request cancellation without blocking the UI event loop.
        self.cancel_requested = True
        if self.run_id is not None:
            event.app.create_background_task(
                asyncio.to_thread(self._stop_run, self.run_id)
            )

    async def _run_prompt(
        self, prompt: str, app_spec: str, fab_hash: str | None
    ) -> None:
        """Run one blocking chat request outside the UI event loop."""
        try:
            await asyncio.to_thread(self._run_prompt_sync, prompt, app_spec, fab_hash)
        except click.ClickException as exc:
            self._append_transcript("class:error", f"Error: {exc.format_message()}\n\n")
        finally:
            finalized_markdown = False
            for entry in self.transcript:
                if isinstance(entry, MarkdownBlock) and _finalize_markdown_block(entry):
                    finalized_markdown = True
            if finalized_markdown:
                self.transcript_revision += 1
            self.run_id = None
            self.busy = False
            self.cancel_requested = False
            self.status = ""
            self.application.layout.focus(self.input_buffer)
            self.application.invalidate()

    def _run_prompt_sync(
        self, prompt: str, app_spec: str, fab_hash: str | None
    ) -> None:
        """Start and stream one Flower AgentApp run."""
        # Start a run in the current conversation series.
        self.run_id, self.series_id = start_chat_run(
            self.stub, prompt, self.federation, self.series_id, app_spec, fab_hash
        )

        if self.cancel_requested:
            self._stop_run(self.run_id)
            return

        terminal_event_seen = False
        response_start = len(self.transcript)
        reasoning_block: _DetailsBlock | None = None
        markdown_block: MarkdownBlock | None = None
        web_search_blocks: dict[str, _DetailsBlock] = {}
        req_events = StreamRunEventsRequest(run_id=self.run_id)
        # Append streamed response content until the run reaches a terminal event.
        with flwr_cli_grpc_exc_handler():
            for res_events in self.stub.StreamRunEvents(req_events):
                event_type, payload = parse_task_event(res_events.task_event)
                if event_type == CHAT_TEXT_DELTA_EVENT:
                    delta = payload.get("delta")
                    if isinstance(delta, str):
                        # Start a new block after tool output so transcript entries
                        # remain in the same order as their streamed events.
                        if (
                            markdown_block is None
                            or self.transcript[-1] is not markdown_block
                        ):
                            self.status = ""
                            markdown_block = MarkdownBlock()
                            self.transcript.append(markdown_block)
                        markdown_block.body += delta
                        self.transcript_revision += 1
                        self.application.invalidate()
                elif event_type in {
                    CHAT_REASONING_DELTA_EVENT,
                    CHAT_TOOL_CALL_STARTED_EVENT,
                    CHAT_TOOL_CALL_COMPLETED_EVENT,
                }:
                    reasoning_block = self._handle_details_event(
                        event_type,
                        payload,
                        response_start,
                        reasoning_block,
                        web_search_blocks,
                    )
                    if (
                        markdown_block is not None
                        and self.transcript[-1] is not markdown_block
                    ):
                        markdown_block.finalized = True
                        markdown_block = None
                elif event_type in CHAT_FAILURE_EVENTS:
                    raise click.ClickException(format_failure_event(payload))
                elif event_type in CHAT_TERMINAL_EVENTS:
                    terminal_event_seen = True

        if not terminal_event_seen and not self.cancel_requested:
            raise click.ClickException(
                "Chat run ended before the agent response completed."
            )

    def _handle_details_event(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        event_type: str,
        payload: JSONObject,
        response_start: int,
        reasoning_block: _DetailsBlock | None,
        web_search_blocks: dict[str, _DetailsBlock],
    ) -> _DetailsBlock | None:
        """Append streamed reasoning or web-search details."""
        if event_type == CHAT_REASONING_DELTA_EVENT:
            delta = payload.get("delta")
            if not isinstance(delta, str) or not delta:
                return reasoning_block
            if reasoning_block is None:
                reasoning_block = _DetailsBlock("Reasoning")
                self.transcript.insert(response_start, reasoning_block)
            reasoning_block.body += delta
        elif (
            event_type == CHAT_TOOL_CALL_STARTED_EVENT
            and payload.get("connector_ref") == CHAT_WEB_SEARCH_CONNECTOR_REF
        ):
            # Reserve reasoning above web search before either block is populated.
            if reasoning_block is None:
                reasoning_block = _DetailsBlock("Reasoning")
                self.transcript.insert(response_start, reasoning_block)
            tool_call_id = cast(str, payload["tool_call_id"])
            block = _DetailsBlock("Web search")
            web_search_blocks[tool_call_id] = block
            self.transcript.append(block)
        elif (
            event_type == CHAT_TOOL_CALL_COMPLETED_EVENT
            and payload.get("connector_ref") == CHAT_WEB_SEARCH_CONNECTOR_REF
        ):
            tool_call_id = cast(str, payload["tool_call_id"])
            output = cast(JSONObject, payload["output"])
            results = cast(list[object], output["results"])
            result_lines: list[str] = []
            for result in results:
                if not isinstance(result, dict):
                    continue
                title = result.get("title")
                url = result.get("url")
                if not isinstance(title, str) or not isinstance(url, str):
                    continue
                lines = [f"{len(result_lines) + 1}. {title}", f"   {url}"]
                snippet = result.get("snippet")
                if isinstance(snippet, str) and snippet:
                    lines.append(f"   {snippet}")
                result_lines.append("\n".join(lines))
            web_search_blocks[tool_call_id].body = "\n\n".join(result_lines)
        else:
            return reasoning_block

        self.transcript_revision += 1
        self.application.invalidate()
        return reasoning_block

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
        self.transcript_revision += 1
        self.application.invalidate()

    def _clear_transcript(self) -> None:
        """Clear the transcript and reset its scroll position."""
        self.transcript.clear()
        self.follow_transcript = True
        self.transcript_revision += 1
        self.application.invalidate()

    def _append_user_message(self, prompt: str) -> None:
        """Append a full-width highlighted user message."""
        # Store logical lines; rendering handles wrapping and row padding.
        for line_index, line in enumerate(prompt.split("\n")):
            prefix = (
                CHAT_USER_PROMPT
                if line_index == 0
                else " " * get_cwidth(CHAT_USER_PROMPT)
            )
            self.transcript.append(("class:user.message", f"{prefix}{line}\n"))
        self.transcript.append(("", "\n"))
        self.transcript_revision += 1
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

    def _render_agent_name(self) -> StyleAndTextTuples:
        """Return the selected agent label with the active federation."""
        return [("class:agent.name", f" ✿ {self.agent_name} · {self.federation} ")]

    def _render_transcript(self) -> StyleAndTextTuples:
        """Return transcript text wrapped to the current terminal width."""
        # Detect manual scrolling against the previous rendered transcript.
        render_info = self.transcript_window.render_info
        if render_info is not None:
            bottom_scroll = max(
                0, render_info.content_height - render_info.window_height
            )
            self.follow_transcript = (
                self.transcript_window.vertical_scroll >= bottom_scroll
            )

        width = self._get_terminal_width()
        cache_key = (self.transcript_revision, width)
        # Rewrap only after a transcript change or terminal resize.
        if cache_key != self.wrapped_transcript_key:
            fragments: StyleAndTextTuples = []
            for entry in self.transcript:
                if isinstance(entry, _DetailsBlock):
                    fragments.extend(self._render_details_block(entry, width))
                elif isinstance(entry, HistoryBlock):
                    fragments.extend(render_history(entry, width))
                elif isinstance(entry, MarkdownBlock):
                    fragments.extend(render_markdown(entry, width))
                else:
                    fragments.append(entry)
            self.wrapped_transcript = _wrap_transcript_fragments(fragments, width)
            self.wrapped_transcript_key = cache_key
        return self.wrapped_transcript

    def _render_details_block(
        self, block: _DetailsBlock, width: int
    ) -> StyleAndTextTuples:
        """Render one clickable transcript details block."""

        def _toggle(mouse_event: MouseEvent) -> object:
            if mouse_event.event_type != MouseEventType.MOUSE_UP:
                return NotImplemented
            block.expanded = not block.expanded
            self.transcript_revision += 1
            self.application.invalidate()
            return None

        marker = "▾" if block.expanded else "▸"
        header = f" {marker} {block.title}"
        header += " " * max(0, width - get_cwidth(header))
        fragments: StyleAndTextTuples = [
            ("class:details.header", f"{header}\n", _toggle)
        ]
        if block.expanded and block.body:
            body = "\n".join(f"   {line}" for line in block.body.splitlines())
            fragments.append(("class:details.body", f"{body}\n"))
        fragments.append(("", "\n"))
        return fragments

    def _transcript_cursor(self) -> Point:
        """Keep the transcript scrolled to its last line."""
        if self.history_block is not None:
            selected_line = 0
            for fragment in self.wrapped_transcript:
                if fragment[0] == "class:history.selected":
                    return Point(x=0, y=selected_line)
                selected_line += fragment[1].count("\n")

        # Cursor rows must match the manually wrapped transcript lines.
        wrapped_text = "".join(fragment[1] for fragment in self.wrapped_transcript)
        lines = wrapped_text.split("\n")
        last_line_index = len(lines) - 1
        if not self.follow_transcript:
            # Preserve manual scrolling and clamp stale offsets after resizing.
            return Point(
                x=0,
                y=min(self.transcript_window.vertical_scroll, last_line_index),
            )

        # Follow the newest transcript line while the view remains at the bottom.
        return Point(x=0, y=last_line_index)


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


def fetch_chat_agents(
    auth_plugin: CliAuthPlugin, federation: str | None
) -> list[_Agent]:
    """Fetch agents available to the authenticated Flower account."""
    if not isinstance(auth_plugin, OidcCliPlugin) or not auth_plugin.access_token:
        raise click.ClickException("Missing authentication tokens. Please login first.")
    headers = {"Authorization": f"Bearer {auth_plugin.access_token}"}
    url = f"{FLWR_SUPERGRID_API_URL}{CHAT_AGENTS_API_PATH}"
    try:
        response = requests.get(
            url,
            headers=headers,
            params={"federation_id": federation} if federation is not None else None,
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise click.ClickException("Failed to load available agents.") from exc

    return _parse_agents(payload)


def _parse_agents(payload: Any) -> list[_Agent]:
    """Parse the user agents API response into completion entries."""
    if not isinstance(payload, dict) or not isinstance(
        raw_agents := payload.get("agents"), list
    ):
        raise click.ClickException("Invalid response from the agents API.")

    agents: list[_Agent] = []
    for raw_agent in raw_agents:
        if not isinstance(raw_agent, dict):
            raise click.ClickException("Invalid response from the agents API.")
        app_spec = raw_agent.get("app_id")
        display_name = raw_agent.get("display_name")
        description = raw_agent.get("description")
        fab_hash = raw_agent.get("fab_hash")
        if not (
            isinstance(app_spec, str)
            and re.fullmatch(APP_ID_PATTERN, app_spec) is not None
            and isinstance(fab_hash, (str, NoneType))
            and isinstance(display_name, (str, NoneType))
            and isinstance(description, (str, NoneType))
        ):
            raise click.ClickException("Invalid response from the agents API.")
        display_name = display_name or app_spec
        description = description or display_name
        agents.append(_Agent(app_spec, display_name, description, fab_hash))
    return agents


def _extract_agent_selection(prompt: str) -> tuple[str | None, str]:
    """Extract a leading agent app spec from a chat prompt."""
    parts = prompt.split(maxsplit=1)
    if not parts or re.fullmatch(APP_ID_PATTERN, parts[0]) is None:
        return None, prompt
    return parts[0], parts[1] if len(parts) == 2 else ""


def _find_agent(app_spec: str, agents: list[_Agent]) -> _Agent | None:
    """Find an agent by app spec."""
    for agent in agents:
        if agent.app_spec == app_spec:
            return agent
    return None


def format_chat_help() -> str:
    """Return formatted chat command help."""
    command_width = max(len(command) for command in CHAT_COMMANDS)
    lines = ["Available Commands:"]
    for command, description in CHAT_COMMANDS.items():
        lines.append(f"  {command:<{command_width}} {description}")
    return "\n".join(lines) + "\n\n"


def start_chat_run(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    stub: ControlStub,
    prompt: str,
    federation: str | None,
    series_id: int | None,
    app_spec: str = FLOWER_AGENT_APP_ID,
    fab_hash: str | None = None,
) -> tuple[int, int | None]:
    """Start one Flower AgentApp run."""
    req = StartRunRequest(
        app_spec=app_spec,
        override_config=user_config_to_proto({CHAT_AGENT_INPUT_KEY: prompt}),
        federation=federation or "",
    )
    if fab_hash is not None:
        req.fab.CopyFrom(Fab(hash_str=fab_hash))
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
    fragments: StyleAndTextTuples, width: int
) -> StyleAndTextTuples:
    """Wrap formatted transcript fragments to the transcript width."""
    wrapped_fragments: StyleAndTextTuples = []
    current_width = 0
    # Track display-cell width across adjacent styled fragments.
    for fragment in fragments:
        style, text = fragment[:2]
        mouse_handler = fragment[2] if len(fragment) == 3 else None

        def append_fragment(
            fragment_style: str,
            fragment_text: str,
            handler: MouseHandler | None = mouse_handler,
        ) -> None:
            wrapped_fragments.append(
                (fragment_style, fragment_text, handler)
                if handler is not None
                else (fragment_style, fragment_text)
            )

        chunk: list[str] = []
        for char in text:
            if char == "\n":
                # Finish explicit lines and extend highlighted user rows.
                if chunk:
                    append_fragment(style, "".join(chunk))
                    chunk = []
                if style == "class:user.message" and current_width < width:
                    append_fragment(style, " " * (width - current_width))
                append_fragment(style, char)
                current_width = 0
                continue

            char_width = get_cwidth(char)
            if current_width and current_width + char_width > width:
                # Insert a visual line break before exceeding the terminal width.
                if chunk:
                    append_fragment(style, "".join(chunk))
                    chunk = []
                if style == "class:user.message" and current_width < width:
                    append_fragment(style, " " * (width - current_width))
                wrapped_fragments.append(("", "\n"))
                current_width = 0
            chunk.append(char)
            current_width += char_width
        if chunk:
            append_fragment(style, "".join(chunk))
    return wrapped_fragments
