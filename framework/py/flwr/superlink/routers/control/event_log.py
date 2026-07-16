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
"""Control API event logging for FastAPI requests."""

from collections.abc import AsyncIterable, Callable, Iterable, Iterator
from functools import partial
from typing import Protocol, cast

from fastapi import Request
from google.protobuf.message import Message
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import State

from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.event_log.typing import LogEntry
from flwr.supercore.protobuf.routing import _call_handler


class FastAPIEventLogWriterPlugin(Protocol):
    """Write Control API event logs from FastAPI requests."""

    def compose_log_before_event(  # pylint: disable=too-many-arguments
        self,
        request: Message,
        context: Request[State],
        account_info: AccountInfo | None,
        method_name: str,
    ) -> LogEntry:
        """Compose a before-event log entry."""

    def compose_log_after_event(  # pylint: disable=too-many-arguments,R0917
        self,
        request: Message,
        context: Request[State],
        account_info: AccountInfo | None,
        method_name: str,
        response: Message | BaseException | None,
    ) -> LogEntry:
        """Compose an after-event log entry."""

    def write_log(self, log_entry: LogEntry) -> None:
        """Write an event log entry."""


def _write_event(
    event_log_plugin: FastAPIEventLogWriterPlugin,
    compose_log: Callable[[], LogEntry],
) -> None:
    """Compose and write an event log entry."""
    event_log_plugin.write_log(compose_log())


class ControlEventLogger:
    """Write Control API event logs around a protobuf handler call."""

    @staticmethod
    async def call(
        func: Callable[..., object],
        http_request: Request[State],
        proto_request: Message,
        dependency_values: dict[str, object],
    ) -> object:
        """Call a handler and write before and after events when configured."""
        # Event log plugins are optional and resolve from application startup state.
        event_log_plugin = cast(
            FastAPIEventLogWriterPlugin | None,
            getattr(http_request.app.state, "control_event_log_plugin", None),
        )
        if event_log_plugin is None:
            # If no event log isn't configured, call the underlying handler directly
            return await _call_handler(func, proto_request, dependency_values)

        account = cast(AccountInfo | None, dependency_values.get("account"))
        if account is None:
            account = AccountInfo(flwr_aid="", account_name="")

        def compose_after_event(
            response: Message | BaseException | None,
        ) -> Callable[[], LogEntry]:
            """Bind the common fields shared by all after-event entries."""
            return partial(
                event_log_plugin.compose_log_after_event,
                request=proto_request,
                context=http_request,
                account_info=account,
                method_name=http_request.url.path,
                response=response,
            )

        # 1. Write the event recorded before handler execution off the event loop.
        await run_in_threadpool(
            _write_event,
            event_log_plugin,
            partial(
                event_log_plugin.compose_log_before_event,
                request=proto_request,
                context=http_request,
                account_info=account,
                method_name=http_request.url.path,
            ),
        )
        try:
            # 2. Execute the Control handler.
            result = await _call_handler(func, proto_request, dependency_values)
        except BaseException as exc:
            # 3. Record handler failures as after-events before propagating them.
            await run_in_threadpool(
                _write_event,
                event_log_plugin,
                compose_after_event(exc),
            )
            raise

        if isinstance(result, AsyncIterable):
            error = FlowerError(
                ApiErrorCode.INVALID_HANDLER_RESPONSE,
                "Async Control streaming handlers are not supported.",
            )
            await run_in_threadpool(
                _write_event,
                event_log_plugin,
                compose_after_event(error),
            )
            raise error

        if isinstance(result, Iterable):

            def response_wrapper() -> Iterator[Message]:
                response: Message | BaseException | None = None
                try:
                    # Keep the final message for the after-event log.
                    # pylint: disable=use-yield-from
                    for response in cast(Iterable[Message], result):
                        yield response
                except BaseException as exc:
                    response = exc
                    raise
                finally:
                    # 3. Streams finish after iteration, so write the after-event here.
                    # StreamingResponse consumes sync iterables in a worker thread.
                    _write_event(
                        event_log_plugin,
                        compose_after_event(response),
                    )

            return response_wrapper()

        response = cast(Message, result)
        # 3. Unary responses are complete, so write the after-event immediately.
        await run_in_threadpool(
            _write_event,
            event_log_plugin,
            compose_after_event(response),
        )
        return response
