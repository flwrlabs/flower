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
from dataclasses import dataclass

import click
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.utils import get_cwidth

from flwr.app import ConfigRecord
from flwr.common.serde import context_from_proto
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    GetRunSeriesRequest,
    ListRunSeriesRequest,
)
from flwr.proto.control_pb2_grpc import ControlStub
from flwr.proto.message_pb2 import Context as ProtoContext  # pylint: disable=E0611
from flwr.proto.runseries_pb2 import RunSeries  # pylint: disable=E0611

from ..utils import flwr_cli_grpc_exc_handler


@dataclass
class HistoryBlock:
    """Interactive conversation history shown inside the transcript."""

    federation: str
    entries: list[RunSeries]
    selected_index: int = 0


def load_history(stub: ControlStub, federation: str) -> HistoryBlock | None:
    """Load chronological conversation history for a federation."""
    with flwr_cli_grpc_exc_handler():
        response = stub.ListRunSeries(ListRunSeriesRequest(federation_id=federation))
    entries = list(reversed(response.entries))
    if not entries:
        return None
    return HistoryBlock(federation, entries, selected_index=len(entries) - 1)


def load_conversation(
    stub: ControlStub, entry: RunSeries, federation: str
) -> list[tuple[str, str]]:
    """Load displayable messages for one conversation."""
    with flwr_cli_grpc_exc_handler():
        response = stub.GetRunSeries(GetRunSeriesRequest(series_id=entry.series_id))
    if response.series.federation != federation:
        raise click.ClickException(
            f"Conversation {entry.series_id} does not belong to {federation}."
        )
    if not response.HasField("context"):
        return []
    return _parse_conversation_context(response.context)


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


def _parse_conversation_context(  # pylint: disable=too-many-branches
    context_proto: ProtoContext,
) -> list[tuple[str, str]]:
    """Extract displayable user and assistant messages from RunSeries context."""
    context = context_from_proto(context_proto)
    record = context.state.get("items")
    if not isinstance(record, ConfigRecord):
        return []
    raw_items = record.get("json")
    if not isinstance(raw_items, list):
        return []

    messages: list[tuple[str, str]] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, str):
            continue
        try:
            item = json.loads(raw_item)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        role = item.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = item.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            text = "".join(parts)
        else:
            continue
        if text:
            messages.append((role, text))
    return messages


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
