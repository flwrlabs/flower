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
"""History command helpers for Flower Chat."""

import json
from collections.abc import Iterable
from dataclasses import dataclass

import click
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.utils import get_cwidth

from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListRunSeriesEventsRequest,
    ListRunSeriesRequest,
)
from flwr.proto.runseries_pb2 import RunSeries  # pylint: disable=E0611
from flwr.proto.task_pb2 import TaskEvent  # pylint: disable=E0611
from flwr.supercore.control import ControlHttpClient

from ..constant import CHAT_TERMINAL_EVENTS, CHAT_TEXT_DELTA_EVENT
from ..utils import flwr_cli_exc_handler


@dataclass
class HistoryBlock:
    """Interactive conversation history shown inside the transcript."""

    federation: str
    entries: list[RunSeries]
    selected_index: int = 0


def load_history(stub: ControlHttpClient, federation: str) -> HistoryBlock | None:
    """Load chronological conversation history for a federation."""
    with flwr_cli_exc_handler():
        response = stub.ListRunSeries(
            ListRunSeriesRequest(federation_id=federation, is_agent=True)
        )
    # Reverse entries to list them from newest to oldest
    entries = list(reversed(response.entries))
    if not entries:
        return None
    return HistoryBlock(federation, entries, selected_index=len(entries) - 1)


def load_conversation(
    stub: ControlHttpClient, entry: RunSeries, federation: str
) -> list[tuple[str, str]]:
    """Load displayable messages for one conversation."""
    if entry.federation != federation:
        raise click.ClickException(
            f"Conversation {entry.series_id} does not belong to {federation}."
        )
    with flwr_cli_exc_handler():
        response = stub.ListRunSeriesEvents(
            ListRunSeriesEventsRequest(series_id=entry.series_id)
        )
    return _parse_conversation_events(response.events)


def render_history(block: HistoryBlock, width: int) -> StyleAndTextTuples:
    """Render the interactive conversation history selector."""
    fragments: StyleAndTextTuples = [
        ("class:notice", f"Conversation history for {block.federation}:\n")
    ]
    series_width = max(len(str(entry.series_id)) for entry in block.entries)
    for index, entry in enumerate(block.entries):
        marker = "❯" if index == block.selected_index else " "
        row = (
            f" {marker} {entry.series_id:>{series_width}}  "
            f"{entry.description or '(no description)'}"
        )
        row = _truncate_to_width(row, width)
        if index == block.selected_index:
            row += " " * max(0, width - get_cwidth(row))
        style = "class:history.selected" if index == block.selected_index else ""
        fragments.append((style, f"{row}\n"))
    fragments.append(
        (
            "class:notice",
            "\nUp/Down to select · Enter to continue · Esc to cancel\n\n",
        )
    )
    return fragments


def _parse_conversation_events(events: Iterable[TaskEvent]) -> list[tuple[str, str]]:
    """Extract displayable messages from persisted run-series events."""
    messages: list[tuple[str, str]] = []
    assistant_parts: list[str] = []

    def flush_assistant_parts() -> None:
        if assistant_parts:
            messages.append(("assistant", "".join(assistant_parts)))
            assistant_parts.clear()

    for event in events:
        try:
            payload = json.loads(event.data)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue

        event_type = event.event
        if not event_type and isinstance(payload.get("type"), str):
            event_type = payload["type"]
        if event_type == CHAT_TEXT_DELTA_EVENT:
            delta = payload.get("delta")
            if isinstance(delta, str):
                assistant_parts.append(delta)
        elif event_type in CHAT_TERMINAL_EVENTS:
            response_text = _parse_response_text(payload)
            if response_text:
                assistant_parts.clear()
                messages.append(("assistant", response_text))
            else:
                flush_assistant_parts()
        elif event_type == "message":
            role = payload.get("role")
            if role not in {"user", "assistant"}:
                continue
            flush_assistant_parts()
            text = _parse_message_content(payload.get("content"))
            if text:
                messages.append((role, text))

    flush_assistant_parts()
    return messages


def _parse_message_content(content: object) -> str | None:
    """Extract text from an Open Responses message item."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return None


def _parse_response_text(payload: dict[str, object]) -> str | None:
    """Extract final assistant text from a terminal Open Responses event."""
    response = payload.get("response")
    if not isinstance(response, dict):
        return None
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        return output_text
    output = response.get("output")
    if not isinstance(output, list):
        return None
    parts: list[str] = []
    for item in output:
        if (
            not isinstance(item, dict)
            or item.get("type") != "message"
            or item.get("role") != "assistant"
        ):
            continue
        text = _parse_message_content(item.get("content"))
        if text:
            parts.append(text)
    return "".join(parts) or None


def _truncate_to_width(text: str, width: int) -> str:
    """Truncate text to a display-cell width, adding an ellipsis if needed."""
    if get_cwidth(text) <= width:
        return text
    content_width = width - 1
    if content_width <= 0:
        return "…" if width else ""

    chars: list[str] = []
    current_width = 0
    for char in text:
        char_width = get_cwidth(char)
        if current_width + char_width > content_width:
            break
        chars.append(char)
        current_width += char_width
    return f"{''.join(chars)}…"
