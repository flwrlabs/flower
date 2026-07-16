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
"""Tests for Control API FastAPI event logging."""

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from typing import cast

import pytest
from fastapi import FastAPI, Request
from google.protobuf.message import Message
from starlette.datastructures import State

from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListRunsRequest,
    ListRunsResponse,
)
from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.event_log.typing import Actor, Event, LogEntry

from .event_log import ControlEventLogger


class _EventLogWriter:
    """Capture event-log calls from a FastAPI Control request."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Message | BaseException | None]] = []
        self.logs: list[LogEntry] = []

    def compose_log_before_event(  # pylint: disable=too-many-arguments
        self,
        request: Message,
        context: Request[State],
        account_info: AccountInfo | None,
        method_name: str,
    ) -> LogEntry:
        """Capture a before-event call."""
        del context, account_info, method_name
        self.calls.append(("before", request))
        return _log_entry("before")

    def compose_log_after_event(  # pylint: disable=too-many-arguments,R0917
        self,
        request: Message,
        context: Request[State],
        account_info: AccountInfo | None,
        method_name: str,
        response: Message | BaseException | None,
    ) -> LogEntry:
        """Capture an after-event call."""
        del request, context, account_info, method_name
        self.calls.append(("after", response))
        return _log_entry("after")

    def write_log(self, log_entry: LogEntry) -> None:
        """Capture a written entry."""
        self.logs.append(log_entry)


def _log_entry(status: str) -> LogEntry:
    """Return a minimal event-log entry."""
    return LogEntry(
        timestamp=status,
        actor=Actor(actor_id=None, description=None, ip_address=""),
        event=Event(action=status, run_id=None, fab_hash=None),
        status=status,
    )


def _request(event_log_writer: _EventLogWriter) -> Request[State]:
    """Return a Control request configured with the supplied event-log writer."""
    app = FastAPI()
    app.state.control_event_log_plugin = event_log_writer
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/control/list-runs",
            "headers": [],
            "app": app,
        }
    )


def _call(
    handler: Callable[..., object],
    event_log_writer: _EventLogWriter,
) -> object:
    """Call the event logger with a configured request."""
    return asyncio.run(
        ControlEventLogger.call(
            handler,
            _request(event_log_writer),
            ListRunsRequest(),
            {},
        )
    )


def test_unary_response_writes_before_and_after_events() -> None:
    """A unary response is recorded before and after handler execution."""
    event_log_writer = _EventLogWriter()
    response = ListRunsResponse()

    def handler(_: ListRunsRequest) -> ListRunsResponse:
        """Return a unary response."""
        return response

    assert _call(handler, event_log_writer) is response
    assert event_log_writer.calls == [
        ("before", ListRunsRequest()),
        ("after", response),
    ]
    assert [log.status for log in event_log_writer.logs] == ["before", "after"]


def test_unary_exception_writes_after_event() -> None:
    """A unary exception is recorded as the after-event response."""
    event_log_writer = _EventLogWriter()

    def handler(_: ListRunsRequest) -> ListRunsResponse:
        """Raise a handler exception."""
        raise RuntimeError("handler failed")

    with pytest.raises(RuntimeError, match="handler failed"):
        _call(handler, event_log_writer)

    assert event_log_writer.calls[0] == ("before", ListRunsRequest())
    assert isinstance(event_log_writer.calls[1][1], RuntimeError)


def test_stream_writes_after_event_after_consumption() -> None:
    """A synchronous stream is recorded after its final response is consumed."""
    event_log_writer = _EventLogWriter()
    first_response = ListRunsResponse()
    last_response = ListRunsResponse()

    def handler(_: ListRunsRequest) -> Iterator[ListRunsResponse]:
        """Return a two-item stream."""
        yield first_response
        yield last_response

    stream = cast(Iterator[ListRunsResponse], _call(handler, event_log_writer))

    assert list(stream) == [first_response, last_response]
    assert event_log_writer.calls == [
        ("before", ListRunsRequest()),
        ("after", last_response),
    ]


def test_stream_exception_writes_after_event() -> None:
    """A synchronous stream exception is recorded after iteration fails."""
    event_log_writer = _EventLogWriter()

    def handler(_: ListRunsRequest) -> Iterator[ListRunsResponse]:
        """Return a stream that fails after yielding one response."""
        yield ListRunsResponse()
        raise RuntimeError("stream failed")

    stream = cast(Iterator[ListRunsResponse], _call(handler, event_log_writer))

    with pytest.raises(RuntimeError, match="stream failed"):
        list(stream)

    assert event_log_writer.calls[0] == ("before", ListRunsRequest())
    assert isinstance(event_log_writer.calls[1][1], RuntimeError)


def test_async_stream_is_not_supported() -> None:
    """An async stream is rejected and recorded as a failed after-event."""
    event_log_writer = _EventLogWriter()

    async def handler(_: ListRunsRequest) -> AsyncIterator[ListRunsResponse]:
        """Return an unsupported async stream."""
        yield ListRunsResponse()

    with pytest.raises(FlowerError) as error:
        _call(handler, event_log_writer)

    assert error.value.code == ApiErrorCode.INVALID_HANDLER_RESPONSE
    assert event_log_writer.calls[0] == ("before", ListRunsRequest())
    assert isinstance(event_log_writer.calls[1][1], FlowerError)
