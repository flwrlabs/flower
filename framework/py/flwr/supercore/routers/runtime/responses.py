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
"""Open Responses Runtime endpoint backed by asynchronous provider calls.

Inference runs in this API process using FLWR_MODEL_API_ENDPOINT and its
server-side FLWR_MODEL_API_KEY. Usage belongs to the authenticated AgentApp.
A response owns its capacity slot, provider and task watcher until HTTP delivery
ends, including cancellation and the bounded final-error delivery allowance.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from contextlib import aclosing
from functools import partial
from logging import ERROR
from typing import cast

import anyio
import httpx
from anyio import CancelScope
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect
from starlette.types import Message, Receive, Scope, Send

from flwr.common.constant import Status
from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.supercore import log
from flwr.supercore.constant import TaskType
from flwr.supercore.corestate import CoreState
from flwr.supercore.dependencies.runtime import RuntimeStateDependency
from flwr.supercore.model_provider import (
    TERMINAL_EVENTS,
    ModelProviderError,
    iter_model_provider_response,
    validate_model_request,
)
from flwr.supercore.task_process.usage import task_usage_from_open_response
from flwr.supercore.tracing import trace_span
from flwr.supercore.typing import JSONObject
from flwr.supercore.utils import strict_json_dumps

router = APIRouter(prefix="/v1/runtime", tags=["Runtime"])
_SUPPORTED_FIELDS = frozenset(
    {
        "model",
        "input",
        "stream",
        "tools",
        "tool_choice",
        "reasoning",
        "previous_response_id",
        "instructions",
        "max_output_tokens",
        "metadata",
        "text",
    }
)
# pylint: disable-next=fixme
# TODO: Tune this provisional per-process limit with hosted load tests.
_MODEL_CONCURRENCY = 8
_ERROR_DELIVERY_TIMEOUT = 1.0
_POLL_INTERVAL = 0.25
_DEFAULT_MODEL_RESPONSE_TIMEOUT = 900.0
_MODEL_RESPONSE_TIMEOUT_ENV = "FLWR_MODEL_RESPONSE_TIMEOUT"


class _ResponsesError(Exception):
    """An error returned through the Responses HTTP contract."""

    def __init__(self, status_code: int, message: str, code: str) -> None:
        super().__init__(message)
        self.status_code, self.message, self.code = status_code, message, code


@router.post("/responses")
async def create_runtime_response(
    request: Request, state: RuntimeStateDependency
) -> Response:
    """Authenticate, validate and return one directly executed model response."""
    try:
        task = await run_in_threadpool(_authenticate, request, state)
        payload = _normalize_model_request_payload(await _read_request_payload(request))
        return ModelResponse(request, state, task, payload)
    except _ResponsesError as error:
        return _error_response(error)
    except Exception:  # pylint: disable=broad-exception-caught
        log(ERROR, "Runtime Responses request failed unexpectedly", exc_info=True)
        return _error_response(
            _ResponsesError(500, "Internal server error.", "internal_error")
        )


class ModelResponse(Response):
    """Keep provider execution and HTTP delivery within the same cancellation scope."""

    def __init__(
        self, request: Request, state: CoreState, task: Task, payload: JSONObject
    ) -> None:
        super().__init__()
        self.request, self.state, self.task, self.payload = (
            request,
            state,
            task,
            payload,
        )
        self.sequence = 0
        self.capacity: asyncio.Semaphore = getattr(
            request.app.state, "model_response_capacity", None
        ) or asyncio.Semaphore(_MODEL_CONCURRENCY)
        request.app.state.model_response_capacity = self.capacity

    async def _events(self) -> AsyncGenerator[JSONObject, None]:
        usage_recorded = False
        async with aclosing(iter_model_provider_response(self.payload)) as events:
            async for event in events:
                sequence = event.get("sequence_number")
                if isinstance(sequence, int):
                    self.sequence = sequence + 1
                terminal = (
                    self.payload.get("stream") is not True
                    or event.get("type") in TERMINAL_EVENTS
                )
                response = event.get("response", event)
                if terminal and not usage_recorded and isinstance(response, dict):
                    usage = task_usage_from_open_response(
                        response, provider=cast(str, self.payload["model"])
                    )
                    if usage is not None:
                        # The usage write can finish in its thread after cancellation.
                        await anyio.to_thread.run_sync(
                            self.state.add_task_usage,
                            self.task.task_id,
                            usage,
                            abandon_on_cancel=True,
                        )
                        usage_recorded = True
                yield event

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Run and deliver inference while holding one capacity slot."""
        if self.capacity.locked():
            await _error_response(
                _ResponsesError(429, "Model capacity is full.", "rate_limit_exceeded")
            )(scope, receive, send)
            return
        started, complete = False, False
        error: _ResponsesError | None = None

        async def forward(message: Message) -> None:
            nonlocal started, complete
            await send(message)
            if message["type"] == "http.response.start":
                started = True
            if message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                complete = True

        attributes = {
            "flwr.component": "superlink",
            "flwr.run_id": str(self.task.run_id),
            "flwr.task_id": str(self.task.task_id),
            "flwr.model.execution_path": "direct",
        }
        async with self.capacity:
            with trace_span("model.round_trip", attributes=attributes):
                try:
                    with CancelScope() as execution:
                        async with anyio.create_task_group() as group:
                            group.start_soon(
                                _watch_task,
                                self.request,
                                self.state,
                                self.task,
                                execution,
                            )
                            try:
                                with anyio.fail_after(_model_response_timeout()):
                                    await self._deliver(scope, receive, forward)
                            except _ResponsesError as exc:
                                error = exc
                            except (
                                ModelProviderError,
                                httpx.HTTPError,
                                ValueError,
                            ) as exc:
                                error = _ResponsesError(
                                    getattr(exc, "status_code", None) or 502,
                                    str(exc),
                                    "model_provider_error",
                                )
                            except TimeoutError:
                                error = _ResponsesError(
                                    504,
                                    "Model response timed out.",
                                    "model_response_timeout",
                                )
                            except (ClientDisconnect, OSError):
                                return
                            finally:
                                group.cancel_scope.cancel()
                    if execution.cancel_called and not complete:
                        error = _ResponsesError(
                            409, "Model request cancelled.", "model_request_cancelled"
                        )
                except Exception:  # pylint: disable=broad-exception-caught
                    log(ERROR, "Model request failed", exc_info=True)
                    error = _ResponsesError(
                        500, "Internal server error.", "internal_error"
                    )
                if error is not None and not complete:
                    if started and self.payload.get("stream") is not True:
                        # Abort committed JSON so the server closes the connection.
                        raise error
                    with anyio.move_on_after(_ERROR_DELIVERY_TIMEOUT):
                        if started:
                            await send(
                                {
                                    "type": "http.response.body",
                                    "body": _stream_error(
                                        error.message, error.code, self.sequence
                                    ).encode(),
                                    "more_body": False,
                                }
                            )
                        else:
                            await _error_response(error)(scope, receive, send)

    async def _deliver(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Deliver either the JSON response or the provider event stream."""
        async with aclosing(self._events()) as events:
            if self.payload.get("stream") is True:
                frames = (_sse_frame(event) async for event in events)
                async with aclosing(frames):
                    stream = StreamingResponse(
                        frames,
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "X-Accel-Buffering": "no",
                        },
                    )
                    # The task watcher handles disconnects. Keep provider iteration
                    # and cleanup in this task, including on older ASGI versions.
                    await stream.stream_response(send)
            else:
                response = await anext(events)
                _raise_for_failed_response(response)
                await JSONResponse(response)(scope, receive, send)


async def _watch_task(
    request: Request, state: CoreState, task: Task, scope: CancelScope
) -> None:
    """Cancel a quiet provider when its task finishes or the client disconnects."""
    while True:
        tasks = await anyio.to_thread.run_sync(
            partial(state.get_tasks, task_ids=[task.task_id]), abandon_on_cancel=True
        )
        if (
            not tasks
            or tasks[0].status.status == Status.FINISHED
            or await request.is_disconnected()
        ):
            scope.cancel()
            return
        await anyio.sleep(_POLL_INTERVAL)


def _authenticate(request: Request, state: CoreState) -> Task:
    """Authenticate exactly one AgentApp Bearer token."""
    authorization = request.headers.getlist("authorization")
    if len(authorization) != 1:
        raise _ResponsesError(
            401, "Invalid authentication credentials.", "invalid_api_key"
        )

    parts = authorization[0].split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1]:
        raise _ResponsesError(
            401, "Invalid authentication credentials.", "invalid_api_key"
        )

    task = state.get_task_by_token(parts[1])
    if task is None or task.type != TaskType.AGENT_APP:
        raise _ResponsesError(
            401, "Invalid authentication credentials.", "invalid_api_key"
        )
    return task


async def _read_request_payload(request: Request) -> JSONObject:
    """Parse and validate the JSON request object."""
    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise _ResponsesError(
            400, "Request body must be valid JSON.", "invalid_json"
        ) from err

    if not isinstance(payload, dict):
        raise _ResponsesError(
            400, "Request body must be a JSON object.", "invalid_request"
        )
    unsupported = sorted(set(payload) - _SUPPORTED_FIELDS)
    if unsupported:
        fields = ", ".join(unsupported)
        raise _ResponsesError(
            400,
            f"Unsupported request field(s): {fields}.",
            "unsupported_parameter",
        )
    return cast(JSONObject, payload)


def _normalize_model_request_payload(payload: JSONObject) -> JSONObject:
    """Normalize and validate a model request payload."""
    payload = {key: value for key, value in payload.items() if value is not None}
    payload.setdefault("stream", False)
    try:
        validate_model_request(payload)
    except (TypeError, ValueError) as err:
        raise _ResponsesError(400, str(err), "invalid_request") from err
    return payload


def _model_response_timeout() -> float:
    """Return the configured maximum time for receiving a model response."""
    raw_timeout = os.getenv(
        _MODEL_RESPONSE_TIMEOUT_ENV,
        str(_DEFAULT_MODEL_RESPONSE_TIMEOUT),
    )
    try:
        timeout = float(raw_timeout.strip())
    except ValueError:
        timeout = _DEFAULT_MODEL_RESPONSE_TIMEOUT
    return max(1.0, timeout)


def _raise_for_failed_response(response: JSONObject) -> None:
    """Map a structured failed ModelResponse to an HTTP error."""
    if response.get("status") == "failed" or response.get("error") is not None:
        raise _ResponsesError(
            502, _response_error_message(response), "model_provider_error"
        )


def _response_error_message(response: JSONObject) -> str:
    """Extract a safe message from a structured failed response."""
    error = response.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return cast(str, error["message"])
    return "Model request failed."


def _sse_frame(event: JSONObject) -> str:
    """Encode a validated provider event as one SSE frame."""
    return f"event: {event['type']}\ndata: {strict_json_dumps(event, compact=True)}\n\n"


def _stream_error(message: str, code: str, sequence_number: int) -> str:
    """Encode a terminal Responses error event."""
    data: JSONObject = {
        "type": "error",
        "code": code,
        "message": message,
        "param": None,
        "sequence_number": sequence_number,
    }
    return _sse_frame(data)


def _error_response(error: _ResponsesError) -> JSONResponse:
    """Return an OpenAI-compatible JSON error envelope."""
    headers = {"WWW-Authenticate": "Bearer"} if error.status_code == 401 else None
    return JSONResponse(
        status_code=error.status_code,
        headers=headers,
        content={
            "error": {
                "message": error.message,
                "type": (
                    "invalid_request_error"
                    if error.status_code in {400, 401}
                    else "server_error"
                ),
                "param": None,
                "code": error.code,
            }
        },
    )
