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
"""Tests for Runtime tracing transport, completion and failure isolation."""

from functools import partial
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    AcquireTaskRequest,
    AcquireTaskResponse,
    PullTaskInputRequest,
    PullTaskInputResponse,
    PushTaskOutputRequest,
    PushTaskOutputResponse,
)
from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.supercore.constant import TASK_TOKEN_HEADER
from flwr.supercore.dependencies.runtime import get_runtime_state, get_task
from flwr.supercore.interceptors import RuntimeTokenHttpInterceptor
from flwr.supercore.protobuf.client import ProtobufRequestContext
from flwr.supercore.protobuf.constants import PROTOBUF_MEDIA_TYPE
from flwr.supercore.protobuf.translation import ProtobufTranslationMiddleware
from flwr.supercore.routers.runtime.router import router
from flwr.supercore.runtime import RuntimeHttpClient

from . import tracing

_CARRIER = "00-" + "1" * 32 + "-" + "2" * 16 + "-03"


@pytest.fixture(name="app")
def runtime_app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """Provide the real Runtime routes with stubbed business state and handlers."""
    monkeypatch.setenv("FLWR_TRACING_ENABLED", "1")
    state = Mock()
    app = FastAPI()
    app.include_router(router)
    app.state.superexec_auth_secret = None
    app.state.runtime_handlers = Mock()
    app.state.runtime_state_factory = Mock(state=lambda: state)
    app.dependency_overrides[get_runtime_state] = lambda: state
    app.dependency_overrides[get_task] = lambda: Task(task_id=5, run_id=7)
    app.add_middleware(ProtobufTranslationMiddleware)
    return app


def test_direct_request_context_without_persisted_task_parent(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    """Requests keep HTTP context without returning persisted task parents."""
    monkeypatch.setattr(tracing, "current_traceparent", lambda: _CARRIER)
    handlers = app.state.runtime_handlers
    handlers.acquire_task.return_value = AcquireTaskResponse(
        task=Task(task_id=5), token="test-token"
    )
    handlers.pull_task_input.return_value = PullTaskInputResponse(task_id=5)
    interceptor = tracing.RuntimeTracingInterceptor()
    with (
        TestClient(app) as server,
        RuntimeHttpClient.from_server_address(
            "runtime.example",
            True,
            None,
            interceptors=[interceptor, RuntimeTokenHttpInterceptor("test-token")],
        ) as client,
    ):

        def send(request: httpx.Request) -> httpx.Response:
            assert request.headers["traceparent"] == _CARRIER
            assert request.headers[TASK_TOKEN_HEADER] == "test-token"
            response = server.post(
                request.url.path, content=request.content, headers=dict(request.headers)
            )
            assert "traceparent" not in response.headers
            return httpx.Response(
                response.status_code,
                headers=response.headers,
                content=response.content,
                request=request,
            )

        with patch.object(
            client._client, "send", side_effect=send  # pylint: disable=protected-access
        ):
            assert client.AcquireTask(AcquireTaskRequest()).task.task_id == 5
            assert client.PullTaskInput(PullTaskInputRequest()).task_id == 5


def test_tracing_interceptor_failure_preserves_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Telemetry failures preserve responses and do not swallow transport errors."""
    monkeypatch.setattr(
        tracing,
        "current_traceparent",
        Mock(side_effect=RuntimeError("telemetry failure")),
    )
    request = httpx.Request("POST", "http://runtime.example")
    context = ProtobufRequestContext(
        "/flwr.proto.Runtime/PullTaskInput", PullTaskInputRequest(), request
    )
    response = httpx.Response(
        200,
        request=request,
    )
    error = httpx.ConnectError("transport failure")
    send = Mock(side_effect=[response, error])
    interceptor = tracing.RuntimeTracingInterceptor()
    assert interceptor.intercept(context, send) is response
    with pytest.raises(httpx.ConnectError) as caught:
        interceptor.intercept(context, send)
    assert caught.value is error


@pytest.mark.parametrize("failed", [False, True])
def test_completion_tracing_scopes_handler_and_preserves_failure(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI, failed: bool
) -> None:
    """The dependency wraps business work and preserves application failures."""
    manager = MagicMock()
    backend = Mock(trace_span=Mock(return_value=manager))
    monkeypatch.setattr("flwr.supercore.tracing._load_backend", lambda: backend)
    error = ValueError("business failure")

    def finish(*_args: object) -> PushTaskOutputResponse:
        manager.__enter__.assert_called_once()
        manager.__exit__.assert_not_called()
        if failed:
            raise error
        return PushTaskOutputResponse()

    app.state.runtime_handlers.push_task_output.side_effect = finish
    with TestClient(app) as server:
        push = partial(
            server.post,
            "/v1/runtime/push-task-output",
            content=PushTaskOutputRequest().SerializeToString(),
            headers={"content-type": PROTOBUF_MEDIA_TYPE, "traceparent": _CARRIER},
        )
        if failed:
            with pytest.raises(ValueError) as caught:
                push()
            assert caught.value is error
        else:
            assert push().status_code == 200
        manager.__exit__.assert_called_once_with(None, None, None)
        assert backend.trace_span.call_args.kwargs["traceparent"] == _CARRIER
