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

from collections.abc import Callable
from functools import partial
from typing import cast

import grpc
from fastapi import Request
from google.protobuf.message import Message
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import State

from flwr.common.event_log_plugin import EventLogWriterPlugin
from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.event_log.typing import LogEntry
from flwr.supercore.protobuf.routing import _call_handler


class _HttpServicerContext:
    """Expose the gRPC context attributes used by Control event log plugins."""

    def __init__(self, request: Request[State]) -> None:
        self.request = request

    def invocation_metadata(self) -> tuple[tuple[str, str], ...]:
        """Return HTTP headers in the same shape as gRPC metadata."""
        return tuple(self.request.headers.items())

    def peer(self) -> str:
        """Return the request client address."""
        client = self.request.client
        return "" if client is None else f"{client.host}:{client.port}"


def _write_event(
    event_log_plugin: EventLogWriterPlugin,
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
        event_log_plugin = cast(
            EventLogWriterPlugin | None,
            getattr(http_request.app.state, "control_event_log_plugin", None),
        )
        if event_log_plugin is None:
            return await _call_handler(func, proto_request, dependency_values)

        account = cast(AccountInfo | None, dependency_values.get("account"))
        if account is None:
            account = AccountInfo(flwr_aid="", account_name="")
        context = cast(grpc.ServicerContext, _HttpServicerContext(http_request))
        method_name = "/flwr.proto.Control/" + "".join(
            part.capitalize() for part in func.__name__.split("_")
        )
        await run_in_threadpool(
            _write_event,
            event_log_plugin,
            partial(
                event_log_plugin.compose_log_before_event,
                request=proto_request,
                context=context,
                account_info=account,
                method_name=method_name,
            ),
        )
        response: Message | BaseException | None = None
        try:
            response = cast(
                Message, await _call_handler(func, proto_request, dependency_values)
            )
            return response
        except BaseException as exc:
            response = exc
            raise
        finally:
            await run_in_threadpool(
                _write_event,
                event_log_plugin,
                partial(
                    event_log_plugin.compose_log_after_event,
                    request=proto_request,
                    context=context,
                    account_info=account,
                    method_name=method_name,
                    response=response,
                ),
            )
