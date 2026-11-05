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
"""Tests for user-wide stream replay, cancellation, and cursor validation."""

from unittest.mock import Mock

import pytest

from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    StreamEventsRequest,
    StreamEventsResponse,
)
from flwr.server.superlink.linkstate import LinkState
from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.error import FlowerError
from flwr.supercore.protobuf.streaming import ProtobufStreamContext
from flwr.superlink.servicer.control.control_handlers import stream_events


def test_replay_and_disconnect() -> None:
    """Use the caller's identity and stop fetching when the connection closes."""
    state = Mock(spec=LinkState)
    state.get_control_events.return_value = [
        StreamEventsResponse(id=8),
        StreamEventsResponse(id=11),
    ]
    context = ProtobufStreamContext()
    stream = stream_events(
        StreamEventsRequest(after_event_id=7),
        AccountInfo(flwr_aid="alice", account_name="Alice"),
        state,
        context.is_active,
    )
    assert next(stream).heartbeat
    assert [next(stream).id, next(stream).id] == [8, 11]
    context.cancel()
    assert not list(stream)
    state.get_control_events.assert_called_once_with("alice", after_id=7, limit=100)
    context = ProtobufStreamContext()
    state.get_control_events.return_value = [StreamEventsResponse(id=12)]
    stream = stream_events(
        StreamEventsRequest(),
        AccountInfo(flwr_aid="alice", account_name="Alice"),
        state,
        context.is_active,
        last_event_id="11",
    )
    assert next(stream).heartbeat and next(stream).id == 12
    context.cancel()
    assert not list(stream)
    assert state.get_control_events.call_args.kwargs["after_id"] == 11


def test_request_cursor_takes_precedence() -> None:
    """An explicit protobuf cursor overrides Last-Event-ID, including zero."""
    state = Mock(spec=LinkState)
    state.get_control_events.return_value = [StreamEventsResponse(id=1)]
    context = ProtobufStreamContext()
    stream = stream_events(
        StreamEventsRequest(after_event_id=0),
        AccountInfo(flwr_aid="alice", account_name="Alice"),
        state,
        context.is_active,
        last_event_id="99",
    )
    assert next(stream).heartbeat and next(stream).id == 1
    context.cancel()
    state.get_control_events.assert_called_once_with("alice", after_id=0, limit=100)


@pytest.mark.parametrize("cursor", ["invalid", "-1", str(2**63)])
def test_invalid_cursor_fails_before_streaming(cursor: str) -> None:
    """Report invalid cursors as HTTP-translatable errors before headers are sent."""
    with pytest.raises(FlowerError):
        stream_events(
            StreamEventsRequest(),
            AccountInfo(flwr_aid="alice", account_name="Alice"),
            Mock(spec=LinkState),
            last_event_id=cursor,
        )
