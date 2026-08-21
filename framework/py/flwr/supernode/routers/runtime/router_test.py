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
"""Tests for the SuperNode Runtime API router."""

from typing import cast
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from google.protobuf.message import Message
from httpx import Response
from pytest import MonkeyPatch

from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    ClaimTaskRequest,
    ClaimTaskResponse,
    PullTaskInputRequest,
    PullTaskInputResponse,
)
from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.supercore.constant import (
    FLWR_COMPONENT_NAME_METADATA_KEY,
    FLWR_PACKAGE_NAME_METADATA_KEY,
    FLWR_PACKAGE_VERSION_METADATA_KEY,
)
from flwr.supercore.error import ApiErrorCode, http_error_translator
from flwr.supercore.protobuf.constants import PROTOBUF_MEDIA_TYPE
from flwr.supercore.protobuf.translation import (
    PROTOBUF_REQUEST_TYPES,
    ProtobufTranslationMiddleware,
)
from flwr.supercore.servicer.runtime import runtime_handlers as core_runtime_handlers
from flwr.supernode.dependencies.nodestate import get_nodestate
from flwr.supernode.dependencies.task import get_task
from flwr.supernode.nodestate import NodeState
from flwr.supernode.servicer.runtime import runtime_handlers

from .router import router

_SUPEREXEC_PATHS = {
    "/v1/runtime/pull-pending-tasks",
    "/v1/runtime/claim-task",
}


def _create_app(
    state: NodeState,
    *,
    task: Task | None = None,
    superexec_auth_secret: bytes | None = None,
) -> FastAPI:
    """Create a minimal app containing the Runtime API stack."""
    app = FastAPI()
    app.state.superexec_auth_secret = superexec_auth_secret
    app.include_router(router)
    app.add_middleware(ProtobufTranslationMiddleware)
    app.middleware("http")(http_error_translator)
    app.dependency_overrides[get_nodestate] = lambda: state
    if task is not None:
        app.dependency_overrides[get_task] = lambda: task
    return app


def _post(client: TestClient, path: str, request: Message) -> Response:
    """Post a protobuf request to a Runtime route."""
    return cast(
        Response,
        client.post(
            path,
            content=request.SerializeToString(),
            headers={"content-type": PROTOBUF_MEDIA_TYPE},
        ),
    )


def test_runtime_route_rejects_incompatible_version() -> None:
    """Runtime routes should reject peers from a different Flower release."""
    client = TestClient(_create_app(Mock(spec=NodeState)))

    response = client.post(
        "/v1/runtime/claim-task",
        content=ClaimTaskRequest(task_id=123).SerializeToString(),
        headers={
            "content-type": PROTOBUF_MEDIA_TYPE,
            FLWR_PACKAGE_NAME_METADATA_KEY: "flwr",
            FLWR_PACKAGE_VERSION_METADATA_KEY: "0.0.1",
            FLWR_COMPONENT_NAME_METADATA_KEY: "SuperExec",
        },
    )

    assert response.status_code == 412
    assert response.json()["code"] == ApiErrorCode.RUNTIME_VERSION_INCOMPATIBLE


def test_all_runtime_routes_have_protobuf_request_types() -> None:
    """Every Runtime route has exactly one protobuf request type mapping."""
    route_keys = {
        (method, route.path)
        for route in router.routes
        if isinstance(route, APIRoute)
        for method in (route.methods or set())
    }
    runtime_request_types = {
        route_key
        for route_key in PROTOBUF_REQUEST_TYPES
        if route_key[1].startswith("/v1/runtime/")
    }

    assert len(route_keys) == 19
    assert route_keys == runtime_request_types


def test_runtime_routes_declare_expected_security() -> None:
    """Only task-authenticated routes declare the task-token security scheme."""
    schema = _create_app(Mock(spec=NodeState)).openapi()

    for path, path_item in schema["paths"].items():
        security = path_item["post"].get("security", [])
        if path in _SUPEREXEC_PATHS:
            assert security == []
        else:
            assert security == [{"RuntimeTaskToken": []}]


def test_claim_task_delegates_to_shared_handler(monkeypatch: MonkeyPatch) -> None:
    """ClaimTask translates protobuf payloads and calls the shared handler."""
    state = Mock(spec=NodeState)
    expected = ClaimTaskResponse(token="task-token")
    handler = Mock(return_value=expected)
    monkeypatch.setattr(core_runtime_handlers, "claim_task", handler)
    client = TestClient(_create_app(state))
    request = ClaimTaskRequest(task_id=123)

    response = _post(client, "/v1/runtime/claim-task", request)

    assert response.status_code == 200
    assert ClaimTaskResponse.FromString(response.content) == expected
    handler.assert_called_once_with(request, state)


def test_pull_task_input_delegates_with_authenticated_task(
    monkeypatch: MonkeyPatch,
) -> None:
    """Task-authenticated routes pass NodeState and the task to their handler."""
    state = Mock(spec=NodeState)
    task = Task(task_id=123)
    expected = PullTaskInputResponse()
    handler = Mock(return_value=expected)
    monkeypatch.setattr(runtime_handlers, "pull_task_input", handler)
    client = TestClient(_create_app(state, task=task))
    request = PullTaskInputRequest()

    response = _post(client, "/v1/runtime/pull-task-input", request)

    assert response.status_code == 200
    assert PullTaskInputResponse.FromString(response.content) == expected
    handler.assert_called_once_with(request, state, task)


def test_superexec_route_rejects_unsigned_request_when_auth_is_enabled() -> None:
    """SuperExec-authenticated routes reject missing signature headers."""
    state = Mock(spec=NodeState)
    client = TestClient(_create_app(state, superexec_auth_secret=b"superexec-secret"))

    response = _post(client, "/v1/runtime/claim-task", ClaimTaskRequest(task_id=123))

    assert response.status_code == 401
    assert response.json()["code"] == ApiErrorCode.RUNTIME_AUTHENTICATION_FAILED
