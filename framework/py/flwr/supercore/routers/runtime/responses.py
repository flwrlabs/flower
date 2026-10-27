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
"""Open Responses-compatible Runtime endpoint.

``FLWR_RUNTIME_MODEL_DIRECT_MODELS`` is an opt-in, comma-separated model list.
Those models call ``FLWR_MODEL_API_ENDPOINT`` using ``FLWR_MODEL_API_KEY`` in
this API process, with no child MODEL task. Unlisted models retain task execution.
An empty list disables the bypass. The endpoint must be the inference endpoint,
not this Runtime route. Keep these settings and credentials on the server.

Direct calls have eight slots per API process and use the existing provider and
Runtime response timeouts. Full capacity returns 429 rather than switching paths.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncGenerator, Iterator
from contextlib import ExitStack, aclosing, contextmanager
from dataclasses import dataclass
from functools import partial
from logging import ERROR
from time import monotonic
from typing import cast

import anyio
import httpx
from anyio import CancelScope
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect
from starlette.types import Message, Receive, Scope, Send

from flwr.app.constants import DEFAULT_TTL
from flwr.app.message_type import MessageType
from flwr.app.metadata import Metadata
from flwr.common.constant import Status, SubStatus
from flwr.proto.runtime_pb2 import CreateTaskRequest  # pylint: disable=E0611
from flwr.proto.task_pb2 import Task, TaskEvent  # pylint: disable=E0611
from flwr.supercore import log
from flwr.supercore.constant import TaskType
from flwr.supercore.corestate import CoreState
from flwr.supercore.date import now
from flwr.supercore.dependencies.runtime import RuntimeStateDependency
from flwr.supercore.error import FlowerError
from flwr.supercore.json_message.base import make_json_message
from flwr.supercore.json_message.model_message import ModelRequest, ModelResponse
from flwr.supercore.servicer.runtime import runtime_handlers
from flwr.supercore.task_process.model.provider import (
    ModelProviderError,
    iter_model_provider_response,
)
from flwr.supercore.task_process.usage import task_usage_from_open_response
from flwr.supercore.tracing import TraceSpan, trace_span
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
_TERMINAL_EVENTS = frozenset(
    {"error", "response.completed", "response.failed", "response.incomplete"}
)
# pylint: disable-next=fixme
# TODO: Tune this provisional per-process limit with DEV load tests before rollout.
_DIRECT_MODEL_CONCURRENCY = 8
# Allow final errors to reach clients without indefinitely retaining a blocked slot.
_DIRECT_ERROR_DELIVERY_TIMEOUT = 1.0
_POLL_INTERVAL = 0.25
_DEFAULT_MODEL_RESPONSE_TIMEOUT = 900.0
_DEFAULT_MODEL_TASK_LAUNCH_TIMEOUT = 300.0
_MODEL_RESPONSE_TIMEOUT_ENV = "FLWR_MODEL_RESPONSE_TIMEOUT"
_MODEL_TASK_LAUNCH_TIMEOUT_ENV = "FLWR_MODEL_TASK_LAUNCH_TIMEOUT"


@dataclass(frozen=True)
class _Exchange:
    """Identify one AgentApp-to-model task exchange."""

    agent_task_id: int
    model_task_id: int


@dataclass
class _StreamProgress:
    """Retain relay progress for error framing and child-task cleanup."""

    complete: bool = False
    sequence_number: int = 0
    first_text_seen: bool = False


class _ResponsesError(Exception):
    """Represent an error returned through the Responses HTTP contract."""

    def __init__(self, status_code: int, message: str, code: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code


@router.post("/responses")
async def create_runtime_response(
    request: Request,
    state: RuntimeStateDependency,
) -> Response:
    """Create a model response directly or through a child model task."""
    try:
        task = await run_in_threadpool(_authenticate, request, state)
        payload = await _read_request_payload(request)
        model_payload = _normalize_model_request_payload(payload)
        direct_models = os.getenv("FLWR_RUNTIME_MODEL_DIRECT_MODELS", "").split(",")
        if model_payload["model"] in {
            model.strip() for model in direct_models if model.strip()
        }:
            return _DirectResponse(request, state, task, model_payload)
        if model_payload.get("stream") is True:
            return StreamingResponse(
                _stream_response(state, task, model_payload),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        with _trace_model_exchange(task):
            exchange = await run_in_threadpool(
                _start_exchange,
                state,
                task,
                model_payload,
            )
            with trace_span("model.response_wait"):
                response = await _wait_for_response(request, state, exchange)
    except _ResponsesError as err:
        return _error_response(err)
    except Exception as err:  # pylint: disable=broad-exception-caught
        log(ERROR, "Runtime Responses request failed unexpectedly", exc_info=err)
        return _error_response(
            _ResponsesError(500, "Internal server error.", "internal_error")
        )
    return JSONResponse(content=response)


@contextmanager
def _trace_model_exchange(task: Task, path: str = "task") -> Iterator[None]:
    """Measure the Runtime round trip, including asynchronous model waiting."""
    with trace_span(
        "model.round_trip",
        attributes={
            "flwr.component": "superlink",
            "flwr.run_id": str(task.run_id),
            "flwr.task_id": str(task.task_id),
            "flwr.model.execution_path": path,
        },
    ):
        yield


class _DirectResponse(Response):
    """Own a direct provider call for the full HTTP response lifetime."""

    def __init__(
        self,
        request: Request,
        state: CoreState,
        task: Task,
        payload: JSONObject,
    ) -> None:
        super().__init__()
        self.request, self.state, self.task = request, state, task
        self.payload = payload
        self.capacity: asyncio.Semaphore = getattr(
            request.app.state, "direct_model_capacity", None
        ) or asyncio.Semaphore(_DIRECT_MODEL_CONCURRENCY)
        request.app.state.direct_model_capacity = self.capacity
        self.started = False
        self.sequence = 0

    async def _frames(
        self, events: AsyncGenerator[JSONObject, None]
    ) -> AsyncGenerator[str, None]:
        async for event in events:
            name = event.get("type")
            if not isinstance(name, str):
                raise _ResponsesError(
                    502, "Invalid model event.", "invalid_model_event"
                )
            sequence = event.get("sequence_number")
            if isinstance(sequence, int):
                self.sequence = sequence + 1
            yield _sse_frame(
                TaskEvent(event=name, data=strict_json_dumps(event, compact=True))
            )

    async def _events(self, provider: TraceSpan) -> AsyncGenerator[JSONObject, None]:
        started, first_text, usage_recorded = monotonic(), False, False
        async with aclosing(iter_model_provider_response(self.payload)) as events:
            async for event in events:
                if not first_text and event.get("type") == "response.output_text.delta":
                    first_text = True
                    provider.set_attribute(
                        "flwr.model.first_text_ms", (monotonic() - started) * 1000
                    )
                    provider.add_event("provider.first_text")
                response = event.get("response", event)
                if self.payload.get("stream") is True and event.get("type") in {
                    "response.completed",
                    "response.incomplete",
                }:
                    if "response" not in event or not isinstance(response, dict):
                        raise ValueError(
                            "Terminal model event requires a response object."
                        )
                    ModelResponse.validate_payload(response)
                terminal = (
                    self.payload.get("stream") is not True
                    or event.get("type") in _TERMINAL_EVENTS
                )
                if terminal and not usage_recorded and isinstance(response, dict):
                    usage = task_usage_from_open_response(
                        response, provider=cast(str, self.payload["model"])
                    )
                    if usage is not None:
                        # Cancellation stops waiting, not the in-flight usage write.
                        await anyio.to_thread.run_sync(
                            self.state.add_task_usage,
                            self.task.task_id,
                            usage,
                            abandon_on_cancel=True,
                        )
                        usage_recorded = True
                yield event

    async def _execute(self, scope: Scope, receive: Receive, send: Send) -> None:
        attributes = {
            "flwr.run_id": str(self.task.run_id),
            "flwr.task_id": str(self.task.task_id),
            "flwr.model.name": cast(str, self.payload["model"]),
        }
        with (
            _trace_model_exchange(self.task, "direct"),
            trace_span("model.provider", attributes=attributes) as provider,
        ):
            async with aclosing(self._events(provider)) as events:
                if self.payload.get("stream") is True:
                    async with aclosing(self._frames(events)) as frames:
                        await StreamingResponse(
                            frames,
                            media_type="text/event-stream",
                            headers={
                                "Cache-Control": "no-cache",
                                "X-Accel-Buffering": "no",
                            },
                        )(scope, receive, send)
                else:
                    async for item in events:
                        response = item
                    ModelResponse.validate_payload(response)
                    _raise_for_failed_response(response)
                    await JSONResponse(response)(scope, receive, send)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.capacity.locked():
            await _error_response(
                _ResponsesError(
                    429, "Direct model capacity is full.", "rate_limit_exceeded"
                )
            )(scope, receive, send)
            return
        error = None
        completed = False

        async def forward(message: Message) -> None:
            nonlocal completed
            if message["type"] == "http.response.start":
                self.started = True
            if message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                completed = True
            await send(message)

        async with self.capacity:
            with CancelScope() as execution:
                async with anyio.create_task_group() as group:
                    group.start_soon(
                        _watch_direct_task,
                        self.request,
                        self.state,
                        self.task,
                        execution,
                    )
                    try:
                        with anyio.fail_after(_model_response_timeout()):
                            await self._execute(scope, receive, forward)
                    except _ResponsesError as err:
                        error = err
                    except (ModelProviderError, httpx.HTTPError, ValueError) as err:
                        error = _ResponsesError(
                            getattr(err, "status_code", None) or 502,
                            str(err),
                            "model_provider_error",
                        )
                    except TimeoutError:
                        error = _ResponsesError(
                            504, "Model response timed out.", "model_response_timeout"
                        )
                    except (ClientDisconnect, OSError):
                        return
                    except Exception as err:  # pylint: disable=broad-exception-caught
                        log(ERROR, "Direct model request failed", exc_info=err)
                        error = _ResponsesError(
                            500, "Internal server error.", "internal_error"
                        )
                    finally:
                        group.cancel_scope.cancel()
            if completed:
                return
            if execution.cancel_called:
                error = _ResponsesError(
                    409, "Model request cancelled.", "model_request_cancelled"
                )
            if error is not None:
                with anyio.move_on_after(_DIRECT_ERROR_DELIVERY_TIMEOUT):
                    if self.started:
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


async def _watch_direct_task(
    request: Request, state: CoreState, task: Task, scope: CancelScope
) -> None:
    """Stop a quiet provider after task completion or client disconnect."""
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
        ModelRequest.validate_payload(payload)
    except (TypeError, ValueError) as err:
        raise _ResponsesError(400, str(err), "invalid_request") from err
    return payload


def _start_exchange(
    state: CoreState,
    task: Task,
    payload: JSONObject,
) -> _Exchange:
    """Create a child model task and send its request message."""
    model = cast(str, payload["model"])
    try:
        with trace_span(
            "model.create",
            attributes={
                "flwr.component": "superlink",
                "flwr.run_id": str(task.run_id),
                "flwr.parent_task_id": str(task.task_id),
            },
        ) as span:
            response = runtime_handlers.create_task(
                CreateTaskRequest(type=TaskType.MODEL, model_ref=model), state, task
            )
            if response.HasField("task_id"):
                span.set_attribute("flwr.task_id", str(response.task_id))
    except FlowerError as err:
        raise _ResponsesError(
            500, "Model task could not be created.", "model_task_creation_failed"
        ) from err
    if not response.HasField("task_id"):
        raise _ResponsesError(
            500, "Model task could not be created.", "model_task_creation_failed"
        )

    model_task_id = response.task_id
    node_id = state.get_node_id()
    metadata = Metadata(
        run_id=task.run_id,
        message_id="",
        src_node_id=node_id,
        dst_node_id=node_id,
        reply_to_message_id="",
        group_id="",
        created_at=now().timestamp(),
        ttl=DEFAULT_TTL,
        message_type=MessageType.QUERY,
        src_task_id=task.task_id,
        dst_task_id=model_task_id,
    )
    request = make_json_message(ModelRequest, metadata=metadata, payload=payload)
    request.metadata.__dict__["_message_id"] = request.object_id
    with trace_span("model.request_publish"):
        stored = state.store_task_message(request)
    if not stored:
        state.finish_task(
            model_task_id, SubStatus.STOPPED, "Model request was not stored."
        )
        raise _ResponsesError(
            500, "Model request could not be stored.", "model_request_failed"
        )

    return _Exchange(
        agent_task_id=task.task_id,
        model_task_id=model_task_id,
    )


async def _wait_for_response(
    request: Request, state: CoreState, exchange: _Exchange
) -> JSONObject:
    """Wait for and return one correlated model response."""
    started_at = time.monotonic()
    launch_deadline = started_at + _model_task_launch_timeout()
    response_deadline = started_at + _model_response_timeout()
    complete = False
    try:
        while True:
            if await request.is_disconnected():
                raise _ResponsesError(
                    499, "Client disconnected.", "client_disconnected"
                )

            response = await run_in_threadpool(
                _claim_response_or_raise_for_model_task_state,
                state,
                exchange,
                launch_deadline,
                response_deadline,
            )
            if response is not None:
                complete = True
                _raise_for_failed_response(response)
                return response

            await asyncio.sleep(_POLL_INTERVAL)
    finally:
        if not complete:
            await run_in_threadpool(
                _stop_model_task, state, exchange, "Responses request ended early."
            )


async def _stream_response(
    state: CoreState, task: Task, model_payload: JSONObject
) -> AsyncGenerator[str, None]:
    """Create an exchange and relay its events as Server-Sent Events."""
    with _trace_model_exchange(task):
        progress = _StreamProgress()
        exchange: _Exchange | None = None
        try:
            exchange = await run_in_threadpool(
                _start_exchange,
                state,
                task,
                model_payload,
            )
            with trace_span("model.response_wait"), ExitStack() as first_text_wait:
                first_text = first_text_wait.enter_context(
                    trace_span("model.first_text_wait")
                )
                first_text.set_attribute("flwr.model.first_text_seen", False)
                first_text_recorded = False
                async with aclosing(
                    _relay_model_events(state, exchange, progress)
                ) as relay:
                    async for frame in relay:
                        if progress.first_text_seen and not first_text_recorded:
                            first_text.set_attribute("flwr.model.first_text_seen", True)
                            first_text_wait.close()
                            first_text_recorded = True
                        yield frame
        except _ResponsesError as err:
            yield _stream_error(err.message, err.code, progress.sequence_number)
        except Exception as err:  # pylint: disable=broad-exception-caught
            log(ERROR, "Runtime Responses stream failed unexpectedly", exc_info=err)
            yield _stream_error(
                "Internal server error.", "internal_error", progress.sequence_number
            )
        finally:
            if exchange is not None and not progress.complete:
                with CancelScope(shield=True):
                    await run_in_threadpool(
                        _stop_model_task,
                        state,
                        exchange,
                        "Responses stream ended early.",
                    )


async def _relay_model_events(
    state: CoreState, exchange: _Exchange, progress: _StreamProgress
) -> AsyncGenerator[str, None]:
    """Poll one model exchange and relay its ordered SSE frames."""
    cursor: int | None = None
    started_at = time.monotonic()
    launch_deadline = started_at + _model_task_launch_timeout()
    response_deadline = started_at + _model_response_timeout()
    while True:
        # The model task stores all stream events before its final reply.
        response = await run_in_threadpool(
            _claim_response_or_raise_for_model_task_state,
            state,
            exchange,
            launch_deadline,
            response_deadline,
        )
        events = await run_in_threadpool(
            state.get_task_events,
            task_ids=[exchange.model_task_id],
            after_task_event_id=cursor,
        )
        for event in events:
            cursor = event.id
            if event.event == "response.output_text.delta":
                progress.first_text_seen = True
            if event.event in _TERMINAL_EVENTS:
                if response is None:
                    await _wait_for_terminal_reply(
                        state, exchange, launch_deadline, response_deadline
                    )
                progress.complete = True
                yield _sse_frame(event)
                return
            yield _sse_frame(event)
            progress.sequence_number += 1

        if response is not None:
            progress.complete = True
            yield _stream_error(
                (
                    _response_error_message(response)
                    if response.get("status") == "failed"
                    else "Model stream ended before a terminal event."
                ),
                "model_stream_failed",
                progress.sequence_number,
            )
            return

        await asyncio.sleep(_POLL_INTERVAL)


async def _wait_for_terminal_reply(
    state: CoreState,
    exchange: _Exchange,
    launch_deadline: float,
    response_deadline: float,
) -> JSONObject:
    """Consume the final reply before exposing a terminal stream event."""
    while True:
        response = await run_in_threadpool(
            _claim_response_or_raise_for_model_task_state,
            state,
            exchange,
            launch_deadline,
            response_deadline,
        )
        if response is not None:
            return response
        await asyncio.sleep(_POLL_INTERVAL)


def _claim_response(state: CoreState, exchange: _Exchange) -> JSONObject | None:
    """Atomically claim the reply belonging to this exchange."""
    messages = state.get_task_message(
        dst_task_ids=[exchange.agent_task_id],
        src_task_ids=[exchange.model_task_id],
        limit=1,
        order_by="created_at",
    )
    if not messages:
        return None
    try:
        return ModelResponse.from_message(messages[0]).payload
    except ValueError as err:
        raise _ResponsesError(
            502, "Model task returned an invalid response.", "invalid_model_response"
        ) from err


def _claim_response_or_raise_for_model_task_state(
    state: CoreState,
    exchange: _Exchange,
    launch_deadline: float | None = None,
    response_deadline: float | None = None,
) -> JSONObject | None:
    """Claim a reply, or fail when its task cannot produce one."""
    response = _claim_response(state, exchange)
    if response is not None:
        return response

    tasks = state.get_tasks(task_ids=[exchange.model_task_id])
    if tasks and tasks[0].status.status == Status.FINISHED:
        # The reply is stored immediately before the task is marked finished.
        response = _claim_response(state, exchange)
        if response is not None:
            return response
    if not tasks or tasks[0].status.status == Status.FINISHED:
        details = tasks[0].status.details if tasks else ""
        raise _ResponsesError(
            502,
            details or "Model task ended without a response.",
            "model_task_failed",
        )
    if (
        launch_deadline is not None
        and tasks[0].status.status in {Status.PENDING, Status.STARTING}
        and time.monotonic() >= launch_deadline
    ):
        raise _ResponsesError(
            504,
            "Model task was not launched before the configured timeout.",
            "model_task_launch_timeout",
        )
    if response_deadline is not None and time.monotonic() >= response_deadline:
        raise _ResponsesError(
            504,
            "Model response was not received before the configured timeout.",
            "model_response_timeout",
        )
    return None


def _model_task_launch_timeout() -> float:
    """Return the configured maximum time for launching a model task."""
    raw_timeout = os.getenv(
        _MODEL_TASK_LAUNCH_TIMEOUT_ENV,
        str(_DEFAULT_MODEL_TASK_LAUNCH_TIMEOUT),
    )
    try:
        timeout = float(raw_timeout.strip())
    except ValueError:
        timeout = _DEFAULT_MODEL_TASK_LAUNCH_TIMEOUT
    return max(1.0, timeout)


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


def _stop_model_task(state: CoreState, exchange: _Exchange, details: str) -> None:
    """Stop an unfinished model task and drain an already-arrived reply."""
    state.finish_task(exchange.model_task_id, SubStatus.STOPPED, details)
    try:
        _claim_response(state, exchange)
    except _ResponsesError:
        # Cleanup is best-effort; the original request has already ended.
        pass


def _sse_frame(event: TaskEvent) -> str:
    """Encode one stored task event as an SSE frame."""
    if "\r" in event.event or "\n" in event.event:
        raise _ResponsesError(
            502,
            "Model stream returned an invalid event name.",
            "invalid_model_event",
        )
    data_fields = "".join(
        f"data: {line}\n" for line in (event.data.splitlines() or [""])
    )
    return f"event: {event.event}\n{data_fields}\n"


def _stream_error(message: str, code: str, sequence_number: int) -> str:
    """Encode a terminal Responses error event."""
    data: JSONObject = {
        "type": "error",
        "code": code,
        "message": message,
        "param": None,
        "sequence_number": sequence_number,
    }
    return f"event: error\ndata: {strict_json_dumps(data, compact=True)}\n\n"


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
