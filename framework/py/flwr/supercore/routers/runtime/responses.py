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
Upstream SSE bytes are forwarded unchanged. Standard responses own HTTP delivery,
with a request deadline and cancellation when the AgentApp task ends.
"""

from __future__ import annotations

import json
import os
from contextlib import AsyncExitStack, aclosing
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
from starlette.types import Receive, Scope, Send

from flwr.common.constant import Status
from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.supercore import log
from flwr.supercore.constant import TaskType
from flwr.supercore.corestate import CoreState
from flwr.supercore.dependencies.runtime import RuntimeStateDependency
from flwr.supercore.model_provider import (
    ModelProviderError,
    model_provider_response,
    stream_model_provider_response,
    validate_model_request,
    validate_model_response,
)
from flwr.supercore.task_process.usage import task_usage_from_open_response
from flwr.supercore.tracing import TraceSpan, trace_span
from flwr.supercore.typing import JSONObject

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
_POLL_INTERVAL = 1.0
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
        return ModelResponse(state, task, payload)
    except _ResponsesError as error:
        return _error_response(error)
    except Exception:  # pylint: disable=broad-exception-caught
        log(ERROR, "Runtime Responses request failed unexpectedly", exc_info=True)
        return _error_response(
            _ResponsesError(500, "Internal server error.", "internal_error")
        )


class ModelResponse(Response):
    """Delegate HTTP delivery while bounding inference and watching task lifetime."""

    def __init__(self, state: CoreState, task: Task, payload: JSONObject) -> None:
        super().__init__()
        self.state, self.task, self.payload = state, task, payload

    async def _record_usage(self, response: JSONObject) -> None:
        usage = task_usage_from_open_response(
            response, provider=cast(str, self.payload["model"])
        )
        if usage is not None:
            try:
                await anyio.to_thread.run_sync(
                    self.state.add_task_usage,
                    self.task.task_id,
                    usage,
                    abandon_on_cancel=True,
                )
            except Exception as error:  # pylint: disable=broad-exception-caught
                log(
                    ERROR,
                    "Failed to record usage for task %s",
                    self.task.task_id,
                    exc_info=True,
                )
                raise _ResponsesError(
                    500, "Internal server error.", "internal_error"
                ) from error

    async def _prepare_response(
        self, resources: AsyncExitStack, span: TraceSpan, started: float
    ) -> Response:
        upstream = await resources.enter_async_context(
            model_provider_response(self.payload)
        )
        if self.payload.get("stream") is True:
            body = await resources.enter_async_context(
                aclosing(
                    stream_model_provider_response(
                        upstream, self._record_usage, span, started
                    )
                )
            )
            return StreamingResponse(
                body,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        await upstream.aread()
        body_json = validate_model_response(upstream.json())
        _raise_for_failed_response(body_json)
        await self._record_usage(body_json)
        return JSONResponse(body_json)

    async def _deliver(self, scope: Scope, receive: Receive, send: Send) -> None:
        attributes = {
            "flwr.component": "superlink",
            "flwr.run_id": str(self.task.run_id),
            "flwr.task_id": str(self.task.task_id),
            "flwr.model.execution_path": "direct",
        }
        with (
            trace_span("model.round_trip", attributes=attributes),
            trace_span(
                "model.provider",
                attributes={"flwr.model.name": str(self.payload["model"])},
            ) as span,
        ):
            started = monotonic()
            deadline = anyio.current_time() + _model_response_timeout()
            async with AsyncExitStack() as resources:
                try:
                    # Preparation and delivery share one overall deadline.
                    with anyio.fail_after(max(0, deadline - anyio.current_time())):
                        response = await self._prepare_response(
                            resources, span, started
                        )
                except _ResponsesError as error:
                    response = _error_response(error)
                except (ModelProviderError, httpx.HTTPError, ValueError) as error:
                    response = _error_response(
                        _ResponsesError(
                            getattr(error, "status_code", None) or 502,
                            str(error),
                            "model_provider_error",
                        )
                    )
                except RuntimeError:
                    log(ERROR, "Model response preparation failed", exc_info=True)
                    response = _error_response(
                        _ResponsesError(500, "Internal server error.", "internal_error")
                    )
                try:
                    # All delivery shares the request deadline, including errors.
                    with anyio.fail_after(max(0, deadline - anyio.current_time())):
                        await response(scope, receive, send)
                finally:
                    with CancelScope(shield=True):
                        await resources.aclose()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Close upstream resources on disconnect, task termination or deadline."""
        async with anyio.create_task_group() as group:
            group.start_soon(self._watch_task, group.cancel_scope)
            try:
                await self._deliver(scope, receive, send)
            except (ClientDisconnect, ConnectionError):
                pass
            finally:
                group.cancel_scope.cancel()

    async def _watch_task(self, scope: CancelScope) -> None:
        """Cancel inference when its task finishes."""
        # pylint: disable-next=fixme
        # TODO: Replace polling with task-termination events when available.
        while True:
            tasks = await anyio.to_thread.run_sync(
                partial(self.state.get_tasks, task_ids=[self.task.task_id]),
                abandon_on_cancel=True,
            )
            if not tasks or tasks[0].status.status == Status.FINISHED:
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
