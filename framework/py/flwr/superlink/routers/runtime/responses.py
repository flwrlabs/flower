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
"""Open Responses-compatible Runtime endpoint."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Annotated, cast

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from flwr.common.constant import Status, SubStatus
from flwr.proto.runtime_pb2 import CreateTaskRequest  # pylint: disable=E0611
from flwr.proto.task_pb2 import Task, TaskEvent  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkState
from flwr.supercore.constant import TaskType
from flwr.supercore.error import FlowerError
from flwr.supercore.json_message.model_message import ModelRequest, ModelResponse
from flwr.supercore.servicer.runtime import runtime_handlers
from flwr.supercore.typing import JSONObject
from flwr.supercore.utils import strict_json_dumps
from flwr.superlink.dependencies.linkstate import get_linkstate

router = APIRouter(prefix="/v1/runtime", tags=["Runtime"])

LinkStateDependency = Annotated[LinkState, Depends(get_linkstate)]

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
_POLL_INTERVAL = 0.25
_DEFAULT_MODEL_TASK_LAUNCH_TIMEOUT = 300.0
_MODEL_TASK_LAUNCH_TIMEOUT_ENV = "FLWR_MODEL_TASK_LAUNCH_TIMEOUT"


@dataclass(frozen=True)
class _Exchange:
    """Identify one AgentApp-to-model task exchange."""

    agent_task_id: int
    model_task_id: int
    run_id: int


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
    state: LinkStateDependency,
) -> Response:
    """Create a model response through a child model task."""
    try:
        task = await run_in_threadpool(_authenticate, request, state)
        payload = await _read_request_payload(request)
        model_request = _model_request_from_payload(payload)
        exchange = await run_in_threadpool(_start_exchange, state, task, model_request)
    except _ResponsesError as err:
        return _error_response(err)

    if model_request.payload.get("stream") is True:
        return StreamingResponse(
            _stream_response(state, exchange),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        response = await _wait_for_response(request, state, exchange)
    except _ResponsesError as err:
        return _error_response(err)
    return JSONResponse(content=response)


def _authenticate(request: Request, state: LinkState) -> Task:
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


def _model_request_from_payload(payload: JSONObject) -> ModelRequest:
    """Build and validate the task-routed model request."""
    try:
        return ModelRequest.from_payload(dst_task_id=0, payload=payload)
    except (TypeError, ValueError) as err:
        raise _ResponsesError(400, str(err), "invalid_request") from err


def _start_exchange(
    state: LinkState,
    task: Task,
    request: ModelRequest,
) -> _Exchange:
    """Create a child model task and send its request message."""
    model = cast(str, request.payload["model"])
    try:
        response = runtime_handlers.create_task(
            CreateTaskRequest(type=TaskType.MODEL, model_ref=model), state, task
        )
    except FlowerError as err:
        raise _ResponsesError(
            500, "Model task could not be created.", "model_task_creation_failed"
        ) from err
    if not response.HasField("task_id"):
        raise _ResponsesError(
            500, "Model task could not be created.", "model_task_creation_failed"
        )

    model_task_id = response.task_id
    request.metadata.dst_task_id = model_task_id
    request.metadata.__dict__["_run_id"] = task.run_id
    request.metadata.src_task_id = task.task_id
    request.metadata.__dict__["_message_id"] = request.object_id
    if not state.store_task_message(request):
        state.finish_task(
            model_task_id, SubStatus.STOPPED, "Model request was not stored."
        )
        raise _ResponsesError(
            500, "Model request could not be stored.", "model_request_failed"
        )

    return _Exchange(
        agent_task_id=task.task_id,
        model_task_id=model_task_id,
        run_id=task.run_id,
    )


async def _wait_for_response(
    request: Request, state: LinkState, exchange: _Exchange
) -> JSONObject:
    """Wait for and return one correlated model response."""
    launch_deadline = time.monotonic() + _model_task_launch_timeout()
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
    state: LinkState, exchange: _Exchange
) -> AsyncGenerator[str, None]:
    """Relay correlated child-task events as Server-Sent Events."""
    launch_deadline = time.monotonic() + _model_task_launch_timeout()
    cursor: int | None = None
    complete = False
    response: JSONObject | None = None
    try:
        while True:
            events = await run_in_threadpool(
                state.get_task_events,
                run_id=exchange.run_id,
                task_ids=[exchange.model_task_id],
                after_task_event_id=cursor,
            )
            for event in events:
                cursor = event.id
                if event.event in _TERMINAL_EVENTS:
                    if response is None:
                        await _wait_for_terminal_reply(state, exchange)
                    complete = True
                    yield _sse_frame(event)
                    return
                yield _sse_frame(event)

            if response is not None:
                complete = True
                yield _stream_error(
                    (
                        _response_error_message(response)
                        if response.get("status") == "failed"
                        else "Model stream ended before a terminal event."
                    ),
                    "model_stream_failed",
                )
                return

            response = await run_in_threadpool(
                _claim_response_or_raise_for_model_task_state,
                state,
                exchange,
                launch_deadline,
            )
            if response is None:
                await asyncio.sleep(_POLL_INTERVAL)
    except _ResponsesError as err:
        yield _stream_error(err.message, err.code)
    finally:
        if not complete:
            await run_in_threadpool(
                _stop_model_task, state, exchange, "Responses stream ended early."
            )


async def _wait_for_terminal_reply(
    state: LinkState,
    exchange: _Exchange,
) -> JSONObject:
    """Consume the final reply before exposing a terminal stream event."""
    while True:
        response = await run_in_threadpool(
            _claim_response_or_raise_for_model_task_state, state, exchange
        )
        if response is not None:
            return response
        await asyncio.sleep(_POLL_INTERVAL)


def _claim_response(state: LinkState, exchange: _Exchange) -> JSONObject | None:
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
    state: LinkState,
    exchange: _Exchange,
    launch_deadline: float | None = None,
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


def _stop_model_task(state: LinkState, exchange: _Exchange, details: str) -> None:
    """Stop an unfinished model task and drain an already-arrived reply."""
    state.finish_task(exchange.model_task_id, SubStatus.STOPPED, details)
    try:
        _claim_response(state, exchange)
    except _ResponsesError:
        # Cleanup is best-effort; the original request has already ended.
        pass


def _sse_frame(event: TaskEvent) -> str:
    """Encode one stored task event as an SSE frame."""
    return f"event: {event.event}\ndata: {event.data}\n\n"


def _stream_error(message: str, code: str) -> str:
    """Encode a terminal Responses error event."""
    data: JSONObject = {
        "type": "error",
        "error": {"type": "server_error", "code": code, "message": message},
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
