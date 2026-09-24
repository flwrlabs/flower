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
from threading import Event, Lock
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
    ClaimTaskRequest,
    ClaimTaskResponse,
    GetNodesRequest,
    GetNodesResponse,
    GetRunSeriesEventsRequest,
    GetRunSeriesEventsResponse,
    PullAndClaimTaskRequest,
    PullAndClaimTaskResponse,
    PullPendingTasksRequest,
    PullPendingTasksResponse,
)
from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkState
from flwr.server.superlink.linkstate.in_memory_linkstate import InMemoryLinkState
from flwr.server.superlink.linkstate.sql_linkstate import SqlLinkState
from flwr.supercore.constant import (
    FLWR_COMPONENT_NAME_METADATA_KEY,
    FLWR_PACKAGE_NAME_METADATA_KEY,
    FLWR_PACKAGE_VERSION_METADATA_KEY,
    NOOP_FEDERATION_ID,
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
    "/v1/runtime/pull-and-claim-task",
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


def test_pull_and_claim_task_delegates_to_link_handler(
    monkeypatch: MonkeyPatch,
) -> None:
    """Combined acquisition uses the SuperLink handler."""
    state = Mock(spec=LinkState)
    expected = PullAndClaimTaskResponse(task=Task(task_id=123), token="task-token")
    handler = Mock(return_value=expected)
    monkeypatch.setattr(runtime_handlers, "pull_and_claim_task", handler)
    client = TestClient(_create_app(state))
    request = PullAndClaimTaskRequest(supported_task_types=["flwr-model"])

    response = _post(client, "/v1/runtime/pull-and-claim-task", request)

    assert response.status_code == 200
    assert PullAndClaimTaskResponse.FromString(response.content) == expected
    handler.assert_called_once_with(request, state)


@pytest.mark.parametrize(
    ("producer", "notification", "combined"),
    [
        ("task", True, False),
        ("run", True, False),
        ("task", False, False),
        ("run", True, True),
    ],
)
def test_pull_pending_tasks_waits_for_task_created_by_another_state(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    producer: str,
    notification: bool,
    combined: bool,
) -> None:
    """A request sees committed work with or without an in-process signal."""
    database_path = str(tmp_path / "runtime.db")
    states = [
        SqlLinkState(
            database_path,
            NoOpFederationManager(),
            ObjectStoreFactory().store(),
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

    # Local signals must beat a long fallback. With signals disabled, the short
    # fallback represents work committed by another Runtime process.
    fallback = 5.0 if notification else 0.2
    route = (
        "/v1/runtime/pull-and-claim-task"
        if combined
        else "/v1/runtime/pull-pending-tasks"
    )
    request = (
        PullAndClaimTaskRequest(
            supported_task_types=[TaskType.SERVER_APP], wait_timeout_ms=3_000
        )
        if combined
        else PullPendingTasksRequest(wait_timeout_ms=3_000)
    )
    with patch("flwr.supercore.routers.runtime.router._TASK_RECHECK_SECONDS", fallback):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_post, client, route, request)
            assert first_empty_read.wait(timeout=2)
            if producer == "task":
                task_id = states[1].create_task(task_type=TaskType.MODEL, run_id=42)
            else:
                run_id = states[1].create_run(
                    "", "", None, {}, NOOP_FEDERATION_ID, None, "", TaskType.SERVER_APP
                )
                task_id = states[1].get_run_info(run_ids=[run_id])[0].primary_task_id
            assert task_id is not None
            response = future.result(timeout=2)

    assert response.status_code == 200
    if combined:
        claimed = PullAndClaimTaskResponse.FromString(response.content)
        assert claimed.task.task_id == task_id
        assert claimed.token
    else:
        assert [
            task.task_id
            for task in PullPendingTasksResponse.FromString(response.content).tasks
        ] == [task_id]


def test_rolled_back_task_does_not_wake_waiters(tmp_path: Path) -> None:
    """A nested task insert only signals after its outer SQL transaction commits."""
    state = SqlLinkState(
        str(tmp_path / "runtime.db"),
        NoOpFederationManager(),
        ObjectStoreFactory().store(),
    )
    state.initialize()

    with patch("flwr.supercore.sql_mixin.notify_task_available") as notify:
        with pytest.raises(RuntimeError, match="roll back"):
            with state.session():
                assert state.create_task(task_type=TaskType.MODEL, run_id=42)
                raise RuntimeError("roll back")

    notify.assert_not_called()
    assert not state.get_tasks(statuses=["pending"])


@pytest.mark.parametrize("producer", ["task", "run"])
def test_in_memory_task_creation_wakes_waiter(
    monkeypatch: MonkeyPatch, producer: str
) -> None:
    """Local in-memory producers wake requests without a timer recheck."""
    state = InMemoryLinkState(NoOpFederationManager(), ObjectStoreFactory().store())
    first_empty_read = Event()
    original_get_tasks = state.get_tasks

    def observe_get_tasks(**kwargs: Any) -> object:
        tasks = original_get_tasks(**kwargs)
        if not tasks:
            first_empty_read.set()
        return tasks

    monkeypatch.setattr(state, "get_tasks", observe_get_tasks)
    client = TestClient(_create_app(state))

    with patch("flwr.supercore.routers.runtime.router._TASK_RECHECK_SECONDS", 5.0):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                _post,
                client,
                "/v1/runtime/pull-pending-tasks",
                PullPendingTasksRequest(wait_timeout_ms=3_000),
            )
            assert first_empty_read.wait(timeout=2)
            if producer == "task":
                task_id = state.create_task(task_type=TaskType.MODEL, run_id=42)
            else:
                run_id = state.create_run(
                    "", "", None, {}, NOOP_FEDERATION_ID, None, "", TaskType.SERVER_APP
                )
                task_id = state.get_run_info(run_ids=[run_id])[0].primary_task_id
            response = future.result(timeout=2)

    assert [
        task.task_id
        for task in PullPendingTasksResponse.FromString(response.content).tasks
    ] == [task_id]


def test_pull_and_claim_wait_returns_empty_after_timeout(
    monkeypatch: MonkeyPatch,
) -> None:
    """Unsupported pending work does not end a bounded claim wait."""
    state = Mock(spec=LinkState)
    state.get_tasks.return_value = [Task(task_id=123, type="flwr-clientapp")]
    monkeypatch.setattr(runtime_handlers, "process_due_automations", Mock())
    request = PullAndClaimTaskRequest(
        supported_task_types=["flwr-model"], wait_timeout_ms=50
    )

    response = _post(
        TestClient(_create_app(state)), "/v1/runtime/pull-and-claim-task", request
    )

    assert response.status_code == 200
    assert not PullAndClaimTaskResponse.FromString(response.content).HasField("task")
    assert state.get_tasks.call_count >= 2
    state.claim_task.assert_not_called()


def test_task_wait_is_capped_on_server(monkeypatch: MonkeyPatch) -> None:
    """An oversized wait request still returns an ordinary empty response."""
    state = Mock(spec=LinkState)
    state.get_tasks.return_value = []
    monkeypatch.setattr(runtime_handlers, "process_due_automations", Mock())

    with patch("flwr.supercore.routers.runtime.router._MAX_TASK_WAIT_MS", 40):
        started = monotonic()
        response = _post(
            TestClient(_create_app(state)),
            "/v1/runtime/pull-pending-tasks",
            PullPendingTasksRequest(wait_timeout_ms=1_000),
        )

    assert monotonic() - started < 0.5
    assert response.status_code == 200
    assert not PullPendingTasksResponse.FromString(response.content).tasks


def test_pull_and_claim_wait_grants_one_task_to_one_waiter(
    monkeypatch: MonkeyPatch,
) -> None:
    """Concurrent waits continue after a lost claim race until timeout."""
    state = Mock(spec=LinkState)
    state.get_tasks.return_value = []
    first_reads = Event()
    reads = 0
    claimed = False
    lock = Lock()

    def get_tasks(*args: object, **kwargs: object) -> list[Task]:
        nonlocal reads
        with lock:
            reads += 1
            if reads == 2:
                first_reads.set()
        return cast(list[Task], state.get_tasks.return_value)

    state.get_tasks.side_effect = get_tasks

    def claim_task(task_id: int) -> str | None:
        nonlocal claimed
        with lock:
            if claimed:
                return None
            claimed = True
            return "task-token"

    state.claim_task.side_effect = claim_task
    monkeypatch.setattr(runtime_handlers, "process_due_automations", Mock())
    client = TestClient(_create_app(state))
    request = PullAndClaimTaskRequest(
        supported_task_types=["flwr-model"], wait_timeout_ms=1_300
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_post, client, "/v1/runtime/pull-and-claim-task", request)
            for _ in range(2)
        ]
        assert first_reads.wait(timeout=2)
        state.get_tasks.return_value = [Task(task_id=123, type="flwr-model")]
        responses = [
            PullAndClaimTaskResponse.FromString(future.result(timeout=3).content)
            for future in futures
        ]

    assert sum(response.HasField("task") for response in responses) == 1


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

    response = _post(
        client, "/v1/runtime/pull-and-claim-task", PullAndClaimTaskRequest()
    )

    assert response.status_code == 401
    assert response.json() == {
        "detail": "Authentication failed.",
        "code": ApiErrorCode.RUNTIME_AUTHENTICATION_FAILED.value,
    }
