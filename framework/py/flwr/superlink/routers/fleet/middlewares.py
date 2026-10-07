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
"""Middleware for the Fleet HTTP API."""

from fastapi import Request
from fastapi.responses import Response
from google.protobuf.message import Message
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from flwr.common.event_log_plugin import EventLogWriterPlugin

from .node_auth import authenticate_node


class FleetNodeAuthMiddleware(BaseHTTPMiddleware):
    """Authenticate Fleet HTTP calls before event logging."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Authenticate recognized Fleet requests."""
        if request.url.path.startswith("/v1/fleet/") and isinstance(
            getattr(request.state, "protobuf_request", None), Message
        ):
            await run_in_threadpool(authenticate_node, request)
        return await call_next(request)


class FleetEventLogMiddleware(BaseHTTPMiddleware):
    """Write event logs around Fleet HTTP calls."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Write an event before and after a Fleet handler call."""
        if not request.url.path.startswith("/v1/fleet/"):
            return await call_next(request)

        plugin: EventLogWriterPlugin | None = getattr(
            request.app.state, "fleet_event_log_plugin", None
        )
        protobuf_request = getattr(request.state, "protobuf_request", None)
        if plugin is None or not isinstance(protobuf_request, Message):
            return await call_next(request)

        def write_before() -> None:
            plugin.write_log(
                plugin.compose_log_before_event(
                    request=protobuf_request,
                    context=request,
                    account_info=None,
                    method_name=request.url.path,
                )
            )

        def write_after(result: Message | BaseException | None) -> None:
            plugin.write_log(
                plugin.compose_log_after_event(
                    request=protobuf_request,
                    context=request,
                    account_info=None,
                    method_name=request.url.path,
                    response=result,
                )
            )

        await run_in_threadpool(write_before)
        try:
            response = await call_next(request)
        except BaseException as exc:
            await run_in_threadpool(write_after, exc)
            raise

        await run_in_threadpool(
            write_after, getattr(request.state, "protobuf_response", None)
        )
        return response
