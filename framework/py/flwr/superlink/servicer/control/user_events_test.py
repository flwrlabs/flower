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
"""Tests for user-event replay and connection lifecycle."""

from unittest.mock import Mock, call, patch

import pytest

from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    StreamEventsRequest,
    StreamEventsResponse,
)
from flwr.server.superlink.linkstate import LinkState
from flwr.supercore.auth.typing import AccountInfo
from flwr.superlink.servicer.control.control_handlers import stream_events

_ACCOUNT = AccountInfo(flwr_aid="alice", account_name="Alice")


@pytest.mark.parametrize(
    ("after_seq", "last_event_id", "expected"),
    [(None, None, 0), (None, "7", 7), (3, "invalid", 3), (0, "7", 0)],
)
def test_stream_cursor_precedence(
    after_seq: int | None, last_event_id: str | None, expected: int
) -> None:
    """Prefer the protobuf cursor, then Last-Event-ID, then zero."""
    state = Mock(spec=LinkState)
    stream = stream_events(
        StreamEventsRequest(after_seq=after_seq),
        _ACCOUNT,
        state,
        lambda: True,
        last_event_id,
    )
    state.get_user_events.assert_called_once_with("alice", after_seq=expected)
    stream.close()


def test_stream_replays_pages_polls_and_stops_on_disconnect() -> None:
    """Forward events and cursor markers, then poll from the last delivered seq."""
    state = Mock(spec=LinkState)
    entries = [StreamEventsResponse(seq=seq) for seq in range(1, 5)]
    entries[0].event.type = "agent.created"
    entries[-1].event.type = "message.created"
    state.get_user_events.side_effect = [entries[:2], entries[2:3], [], entries[3:]]
    active = Mock(return_value=True)
    stream = stream_events(StreamEventsRequest(), _ACCOUNT, state, active)
    with patch("flwr.superlink.servicer.control.control_handlers.time.sleep") as sleep:
        assert [next(stream) for _ in entries] == entries
        active.return_value = False
        assert not list(stream)
    sleep.assert_called_once_with(0.5)
    assert state.get_user_events.call_args_list == [
        call("alice", after_seq=0),
        call("alice", after_seq=2),
        call("alice", after_seq=3),
        call("alice", after_seq=3),
    ]


def test_idle_stream_expires_without_heartbeats() -> None:
    """Close the stream after 270 seconds even when the feed remains empty."""
    state = Mock(spec=LinkState)
    state.get_user_events.return_value = []
    stream = stream_events(StreamEventsRequest(), _ACCOUNT, state, lambda: True)
    with (
        patch(
            "flwr.superlink.servicer.control.control_handlers.time.monotonic",
            side_effect=[0, 0, 270],
        ),
        patch("flwr.superlink.servicer.control.control_handlers.time.sleep") as sleep,
    ):
        assert not list(stream)
    sleep.assert_called_once_with(0.5)
