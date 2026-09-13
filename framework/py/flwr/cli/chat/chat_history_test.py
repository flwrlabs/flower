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
"""Tests for conversation history in the CLI `chat` application."""

import asyncio
import json
from unittest.mock import Mock, patch

from flwr.app import ConfigRecord, Context, RecordDict
from flwr.cli.chat.chat_app import ChatApplication
from flwr.cli.chat.chat_history import HistoryBlock
from flwr.cli.constant import CHAT_DEFAULT_FEDERATION_NAME
from flwr.common.serde import context_to_proto
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    GetRunSeriesRequest,
    GetRunSeriesResponse,
    ListRunSeriesRequest,
    ListRunSeriesResponse,
)
from flwr.proto.federation_pb2 import Federation  # pylint: disable=E0611
from flwr.proto.runseries_pb2 import RunSeries  # pylint: disable=E0611

FEDERATION = f"@flower/{CHAT_DEFAULT_FEDERATION_NAME}"


def _create_chat(stub: Mock) -> ChatApplication:
    with patch.object(ChatApplication, "_create_application", return_value=Mock()):
        return ChatApplication(stub, [Federation(name=FEDERATION)], Mock())


def test_history_widget_restores_selected_context() -> None:
    """History should navigate chronologically and restore the selected context."""
    stub = Mock()
    selected = RunSeries(series_id=6, federation=FEDERATION, description="Saved chat")
    latest = RunSeries(series_id=7, federation=FEDERATION, description="Latest chat")
    stub.ListRunSeries.return_value = ListRunSeriesResponse(entries=[latest, selected])
    items = [
        {"type": "message", "role": "user", "content": "Previous question"},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "**Previous answer**"}],
        },
    ]
    context = Context(
        run_id=10,
        node_id=0,
        node_config={},
        state=RecordDict(
            {"items": ConfigRecord({"json": [json.dumps(item) for item in items]})}
        ),
        run_config={},
        series_id=6,
    )
    stub.GetRunSeries.return_value = GetRunSeriesResponse(
        series=selected, context=context_to_proto(context)
    )
    chat = _create_chat(stub)
    event = Mock()

    assert chat._handle_command(event, "/history")  # pylint: disable=protected-access
    assert chat.history_loading
    chat.input_buffer.text = "Do not submit yet"
    chat._submit_prompt(event)  # pylint: disable=protected-access
    assert event.app.create_background_task.call_count == 1
    asyncio.run(event.app.create_background_task.call_args.args[0])
    assert not chat.history_loading
    assert chat.history_block is not None
    chat._move_history_selection(-1)  # pylint: disable=protected-access
    chat.history_loading = True
    chat._move_history_selection(1)  # pylint: disable=protected-access
    assert chat.history_block.selected_index == 0
    asyncio.run(chat._confirm_history_selection())  # pylint: disable=protected-access
    assert not chat.history_loading
    stub.ListRunSeries.assert_called_once_with(
        ListRunSeriesRequest(federation_id=FEDERATION, is_agent=True)
    )
    stub.GetRunSeries.assert_called_once_with(GetRunSeriesRequest(series_id=6))
    assert chat.series_id == 6
    assert chat.transcript == [
        ("class:user.message", "❯ Previous question\n"),
        ("", "\n"),
        ("", "**Previous answer**\n\n"),
    ]


def test_history_handles_empty_result() -> None:
    """An empty history should produce a clear message."""
    stub = Mock(ListRunSeries=Mock(return_value=ListRunSeriesResponse()))
    chat = _create_chat(stub)
    event = Mock()

    assert chat._handle_command(event, "/history")  # pylint: disable=protected-access
    asyncio.run(event.app.create_background_task.call_args.args[0])
    assert chat.transcript == [
        ("class:notice", f"No conversation history found for {FEDERATION}.\n\n")
    ]


def test_history_ignores_cancelled_selection() -> None:
    """A cancelled history selection should not restore its conversation."""
    stub = Mock()
    entry = RunSeries(series_id=6, federation=FEDERATION)
    chat = _create_chat(stub)
    chat.history_block = HistoryBlock(FEDERATION, [entry])
    chat.transcript.append(chat.history_block)
    chat.history_loading = True

    async def load_after_cancel(*_: object) -> list[tuple[str, str]]:
        chat._close_history_selection()  # pylint: disable=protected-access
        return []

    with patch("flwr.cli.chat.chat_app.asyncio.to_thread", new=load_after_cancel):
        asyncio.run(
            chat._confirm_history_selection()  # pylint: disable=protected-access
        )

    assert chat.series_id is None
    assert not chat.transcript
    assert not chat.history_loading
