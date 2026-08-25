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
"""SuperNode Runtime HTTP authentication integration tests."""

from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from google.protobuf.message import Message
from httpx import Response

from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    PullObjectRequest,
    PullObjectResponse,
)
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    CreateTaskRequest,
    GetNodesRequest,
    PullPendingTasksRequest,
    PullPendingTasksResponse,
)
from flwr.supercore.auth import create_superexec_auth_metadata, derive_auth_secret
from flwr.supercore.constant import TASK_TOKEN_HEADER, TaskType
from flwr.supercore.error import ApiErrorCode, http_error_translator
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.protobuf.constants import PROTOBUF_MEDIA_TYPE
from flwr.supercore.protobuf.translation import ProtobufTranslationMiddleware
from flwr.supernode.dependencies.nodestate import get_nodestate
from flwr.supernode.nodestate import NodeState, NodeStateFactory

from .router import router

_SUPEREXEC_SECRET = b"test-superexec-secret"
_PULL_PENDING_TASKS_METHOD = "/flwr.proto.Runtime/PullPendingTasks"


@pytest.fixture(name="state")
def fixture_state() -> NodeState:
    """Create an in-memory NodeState."""
    return NodeStateFactory(objectstore_factory=ObjectStoreFactory()).state()


def _create_app(state: NodeState, secret: bytes | None) -> FastAPI:
    """Create the Runtime HTTP application with real dependencies."""
    app = FastAPI()
    app.state.superexec_auth_secret = secret
    app.include_router(router)
    app.add_middleware(ProtobufTranslationMiddleware)
    app.middleware("http")(http_error_translator)
    app.dependency_overrides[get_nodestate] = lambda: state
    return app


@pytest.fixture(name="client")
def fixture_client(state: NodeState) -> TestClient:
    """Create a Runtime HTTP client with SuperExec auth enabled."""
    return TestClient(_create_app(state, _SUPEREXEC_SECRET))


@pytest.fixture(name="valid_token")
def fixture_valid_token(state: NodeState) -> str:
    """Claim the ClientApp task and return its token."""
    task_id = state.create_task(task_type=TaskType.CLIENT_APP, run_id=99)
    assert task_id is not None
    token = state.claim_task(task_id)
    assert token is not None
    return token


def _post(
    client: TestClient,
    path: str,
    proto_request: Message,
    *,
    token: str | None = None,
    auth_headers: dict[str, str] | None = None,
) -> Response:
    """Send a protobuf request with optional authentication headers."""
    headers = {"content-type": PROTOBUF_MEDIA_TYPE}
    if token is not None:
        headers[TASK_TOKEN_HEADER] = token
    if auth_headers is not None:
        headers.update(auth_headers)
    return cast(
        Response,
        client.post(
            f"/v1/runtime/{path}",
            content=proto_request.SerializeToString(),
            headers=headers,
        ),
    )


def test_runtime_flower_error_is_translated(
    client: TestClient, valid_token: str
) -> None:
    """Translate a handler FlowerError into its configured HTTP status."""
    response = _post(
        client, "create-task", CreateTaskRequest(type=TaskType.MODEL), token=valid_token
    )

    assert response.status_code == 412
    assert response.json()["code"] == ApiErrorCode.RUNTIME_INVALID_TASK_CREATION_REQUEST


def test_pull_object_denied_without_metadata_token(client: TestClient) -> None:
    """Protected routes should deny requests missing a task token."""
    response = _post(client, "pull-object", PullObjectRequest(object_id="obj-1"))

    assert response.status_code == 401
    assert response.json()["code"] == ApiErrorCode.RUNTIME_AUTHENTICATION_FAILED


def test_pull_object_denied_with_invalid_metadata_token(client: TestClient) -> None:
    """Protected routes should deny requests with an invalid task token."""
    response = _post(
        client,
        "pull-object",
        PullObjectRequest(object_id="obj-2"),
        token="invalid-token",
    )

    assert response.status_code == 401
    assert response.json()["code"] == ApiErrorCode.RUNTIME_AUTHENTICATION_FAILED


def test_pull_object_allows_with_valid_metadata_token(
    client: TestClient, valid_token: str
) -> None:
    """Protected routes should allow requests with a valid task token."""
    response = _post(
        client, "pull-object", PullObjectRequest(object_id="obj-3"), token=valid_token
    )

    assert response.status_code == 200
    assert isinstance(
        PullObjectResponse.FromString(response.content), PullObjectResponse
    )


def test_pull_pending_tasks_denied_without_superexec_metadata(
    client: TestClient,
) -> None:
    """SuperExec routes should deny requests missing signed metadata."""
    response = _post(client, "pull-pending-tasks", PullPendingTasksRequest())

    assert response.status_code == 401
    assert response.json()["code"] == ApiErrorCode.RUNTIME_AUTHENTICATION_FAILED


def test_pull_pending_tasks_allows_with_superexec_metadata(
    client: TestClient,
) -> None:
    """SuperExec routes should allow requests with valid signed metadata."""
    proto_request = PullPendingTasksRequest()
    headers = create_superexec_auth_metadata(
        auth_secret=derive_auth_secret(_SUPEREXEC_SECRET),
        method=_PULL_PENDING_TASKS_METHOD,
        request=proto_request,
    )

    response = _post(client, "pull-pending-tasks", proto_request, auth_headers=headers)

    assert response.status_code == 200
    assert isinstance(
        PullPendingTasksResponse.FromString(response.content), PullPendingTasksResponse
    )


def test_get_nodes_allows_auth_then_returns_permission_denied(
    client: TestClient, valid_token: str
) -> None:
    """GetNodes should authenticate, then reject ClientApp tasks."""
    response = _post(client, "get-nodes", GetNodesRequest(), token=valid_token)

    assert response.status_code == 403
    assert response.json()["code"] == ApiErrorCode.RUNTIME_ENDPOINT_UNAVAILABLE


def test_pull_pending_tasks_allows_without_superexec_metadata(state: NodeState) -> None:
    """No SuperExec signing should be required when auth is disabled."""
    client = TestClient(_create_app(state, None))

    response = _post(client, "pull-pending-tasks", PullPendingTasksRequest())

    assert response.status_code == 200
    assert isinstance(
        PullPendingTasksResponse.FromString(response.content), PullPendingTasksResponse
    )
