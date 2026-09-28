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
"""Tests for the Runtime API router."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from time import monotonic
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from google.protobuf.message import Message
from httpx import Response
from pytest import MonkeyPatch

from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    AcquireTaskRequest,
    AcquireTaskResponse,
    ClaimTaskRequest,
    ClaimTaskResponse,
    GetNodesRequest,
    GetNodesResponse,
    GetRunSeriesEventsRequest,
    GetRunSeriesEventsResponse,
)
from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkState
from flwr.server.superlink.linkstate.sql_linkstate import SqlLinkState
from flwr.supercore.constant import (
    FLWR_COMPONENT_NAME_METADATA_KEY,
    FLWR_PACKAGE_NAME_METADATA_KEY,
    FLWR_PACKAGE_VERSION_METADATA_KEY,
    TaskType,
)
from flwr.supercore.dependencies.runtime import get_runtime_state, get_task
from flwr.supercore.dependencies.runtime_version import RuntimeVersionDependency
from flwr.supercore.error import ApiErrorCode, http_error_translator
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.protobuf.constants import PROTOBUF_MEDIA_TYPE
from flwr.supercore.protobuf.translation import (
    PROTOBUF_REQUEST_TYPES,
    ProtobufTranslationMiddleware,
)
from flwr.supercore.routers.runtime import router
from flwr.supercore.servicer.runtime import runtime_handlers as core_runtime_handlers
from flwr.superlink.federation import NoOpFederationManager
from flwr.superlink.servicer.runtime import runtime_handlers

_SUPEREXEC_PATHS = {
    "/v1/runtime/pull-pending-tasks",
    "/v1/runtime/acquire-task",
    "/v1/runtime/claim-task",
}


def _create_app(
    state: LinkState,
    *,
    task: Task | None = None,
    superexec_auth_secret: bytes | None = None,
) -> FastAPI:
    """Create a minimal app containing the Runtime API stack."""
    app = FastAPI()
    app.state.superexec_auth_secret = superexec_auth_secret
    app.state.runtime_handlers = runtime_handlers
    app.include_router(
        router,
        dependencies=[
            Depends(
                RuntimeVersionDependency(
                    component_name="SuperLink",
                    connection_name="Caller <-> SuperLink Runtime API",
                )
            )
        ],
    )
    app.add_middleware(ProtobufTranslationMiddleware)
    app.middleware("http")(http_error_translator)
    app.dependency_overrides[get_runtime_state] = lambda: state
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
    client = TestClient(_create_app(Mock(spec=LinkState)))

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

    assert len(route_keys) == 21
    assert route_keys == runtime_request_types


def test_runtime_routes_declare_expected_security() -> None:
    """Only task-authenticated routes declare the task-token security scheme."""
    schema = _create_app(Mock(spec=LinkState)).openapi()

    for path, path_item in schema["paths"].items():
        security = path_item["post"].get("security", [])
        if path in _SUPEREXEC_PATHS:
            assert security == []
        else:
            assert security == [{"RuntimeTaskToken": []}]


def test_claim_task_delegates_to_shared_handler(monkeypatch: MonkeyPatch) -> None:
    """ClaimTask translates protobuf payloads and calls the shared handler."""
    state = Mock(spec=LinkState)
    expected = ClaimTaskResponse(token="task-token")
    handler = Mock(return_value=expected)
    monkeypatch.setattr(core_runtime_handlers, "claim_task", handler)
    client = TestClient(_create_app(state))
    request = ClaimTaskRequest(task_id=123)

    response = _post(client, "/v1/runtime/claim-task", request)

    assert response.status_code == 200
    assert ClaimTaskResponse.FromString(response.content) == expected
    handler.assert_called_once_with(request, state)


@pytest.mark.parametrize("notification", [True, False])
def test_acquire_task_waits_for_committed_work(
    tmp_path: Path, monkeypatch: MonkeyPatch, notification: bool
) -> None:
    """A wait wakes on a local commit and finds work without a local signal."""
    # pylint: disable=too-many-locals
    database_path = str(tmp_path / "runtime.db")
    states = [
        SqlLinkState(
            database_path, NoOpFederationManager(), ObjectStoreFactory().store()
        )
        for _ in range(2)
    ]
    for state in states:
        state.initialize()
    first_empty_read = Event()
    original_get_tasks = states[0].get_tasks

    def observe_get_tasks(**kwargs: Any) -> object:
        tasks = original_get_tasks(**kwargs)
        if not tasks:
            first_empty_read.set()
        return tasks

    monkeypatch.setattr(states[0], "get_tasks", observe_get_tasks)
    if not notification:
        monkeypatch.setattr("flwr.supercore.sql_mixin.notify_task_available", Mock())
    client = TestClient(_create_app(states[0]))
    request = AcquireTaskRequest(
        supported_task_types=[TaskType.MODEL], wait_timeout_ms=3_000
    )
    recheck_seconds = 5.0 if notification else 0.2

    with patch(
        "flwr.supercore.routers.runtime.router._TASK_RECHECK_SECONDS",
        recheck_seconds,
    ):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_post, client, "/v1/runtime/acquire-task", request)
            assert first_empty_read.wait(timeout=2)
            task_id = states[1].create_task(task_type=TaskType.MODEL, run_id=42)
            assert task_id is not None
            response = future.result(timeout=2)

    assert response.status_code == 200
    acquired = AcquireTaskResponse.FromString(response.content)
    assert acquired.task.task_id == task_id
    assert acquired.token


def test_acquire_task_wait_is_bounded_and_skips_wait_without_capacity(
    monkeypatch: MonkeyPatch,
) -> None:
    """An oversized wait is capped and an empty capacity returns immediately."""
    state = Mock(spec=LinkState)
    state.get_tasks.return_value = []
    monkeypatch.setattr(runtime_handlers, "process_due_automations", Mock())
    client = TestClient(_create_app(state))

    with patch("flwr.supercore.routers.runtime.router._MAX_TASK_WAIT_MS", 40):
        started = monotonic()
        response = _post(
            client,
            "/v1/runtime/acquire-task",
            AcquireTaskRequest(
                supported_task_types=[TaskType.MODEL], wait_timeout_ms=1_000
            ),
        )
    assert response.status_code == 200
    assert monotonic() - started < 0.5
    assert state.get_tasks.call_count >= 2

    state.get_tasks.reset_mock()
    started = monotonic()
    response = _post(
        client,
        "/v1/runtime/acquire-task",
        AcquireTaskRequest(wait_timeout_ms=1_000),
    )
    assert response.status_code == 200
    assert monotonic() - started < 0.5
    state.get_tasks.assert_not_called()


def test_get_nodes_delegates_with_authenticated_task(
    monkeypatch: MonkeyPatch,
) -> None:
    """Task-authenticated routes pass the resolved task to their handler."""
    state = Mock(spec=LinkState)
    task = Task(task_id=123)
    expected = GetNodesResponse()
    handler = Mock(return_value=expected)
    monkeypatch.setattr(runtime_handlers, "get_nodes", handler)
    client = TestClient(_create_app(state, task=task))
    request = GetNodesRequest()

    response = _post(client, "/v1/runtime/get-nodes", request)

    assert response.status_code == 200
    assert GetNodesResponse.FromString(response.content) == expected
    handler.assert_called_once_with(request, state, task)


def test_get_run_series_events_delegates_with_authenticated_task(
    monkeypatch: MonkeyPatch,
) -> None:
    """Run-series event requests should pass the authenticated task."""
    state = Mock(spec=LinkState)
    task = Task(task_id=123)
    expected = GetRunSeriesEventsResponse()
    handler = Mock(return_value=expected)
    monkeypatch.setattr(runtime_handlers, "get_run_series_events", handler)
    client = TestClient(_create_app(state, task=task))
    request = GetRunSeriesEventsRequest()

    response = _post(client, "/v1/runtime/get-run-series-events", request)

    assert response.status_code == 200
    assert GetRunSeriesEventsResponse.FromString(response.content) == expected
    handler.assert_called_once_with(request, state, task)


def test_superexec_route_rejects_unsigned_request_when_auth_is_enabled() -> None:
    """SuperExec-authenticated routes reject missing signature headers."""
    state = Mock(spec=LinkState)
    client = TestClient(_create_app(state, superexec_auth_secret=b"superexec-secret"))

    response = _post(client, "/v1/runtime/acquire-task", AcquireTaskRequest())

    assert response.status_code == 401
    assert response.json() == {
        "detail": "Authentication failed.",
        "code": ApiErrorCode.RUNTIME_AUTHENTICATION_FAILED.value,
    }
