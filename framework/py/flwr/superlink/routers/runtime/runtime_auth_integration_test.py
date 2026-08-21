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
"""SuperLink Runtime HTTP authentication integration tests."""

from collections.abc import Callable
from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from google.protobuf.message import Message
from httpx import Response

from flwr.proto.log_pb2 import (  # pylint: disable=E0611
    PushLogsRequest,
    PushLogsResponse,
)
from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    ConfirmMessageReceivedRequest,
    PullObjectRequest,
    PushObjectRequest,
)
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    GetConnectorRequest,
    GetConnectorResponse,
    GetNodesRequest,
    GetNodesResponse,
    PullAppMessagesRequest,
    PushAppMessagesRequest,
    SendTaskHeartbeatRequest,
    SendTaskHeartbeatResponse,
)
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory
from flwr.supercore.constant import (
    FLWR_IN_MEMORY_DB_NAME,
    NOOP_FEDERATION_ID,
    TASK_TOKEN_HEADER,
    TaskType,
)
from flwr.supercore.error import ApiErrorCode, http_error_translator
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.protobuf.constants import PROTOBUF_MEDIA_TYPE
from flwr.supercore.protobuf.translation import ProtobufTranslationMiddleware
from flwr.superlink.dependencies.linkstate import get_linkstate
from flwr.superlink.federation import NoOpFederationManager

from .router import router

_SERVERAPP_ONLY_CASES: list[tuple[str, Message]] = [
    ("get-nodes", GetNodesRequest()),
    ("push-messages", PushAppMessagesRequest()),
    ("pull-messages", PullAppMessagesRequest()),
    ("push-object", PushObjectRequest()),
    ("pull-object", PullObjectRequest()),
    ("confirm-message-received", ConfirmMessageReceivedRequest()),
]
_SHARED_CASES: list[tuple[str, Message, Callable[[bytes], Message]]] = [
    (
        "send-task-heartbeat",
        SendTaskHeartbeatRequest(),
        SendTaskHeartbeatResponse.FromString,
    ),
    ("push-logs", PushLogsRequest(logs=["hello"]), PushLogsResponse.FromString),
]


@pytest.fixture(name="state")
def fixture_state() -> LinkState:
    """Create an in-memory LinkState."""
    return LinkStateFactory(
        FLWR_IN_MEMORY_DB_NAME,
        NoOpFederationManager(),
        ObjectStoreFactory(),
    ).state()


@pytest.fixture(name="client")
def fixture_client(state: LinkState) -> TestClient:
    """Create the Runtime HTTP application with real dependencies."""
    app = FastAPI()
    app.state.superexec_auth_secret = b"test-superexec-secret"
    app.include_router(router)
    app.add_middleware(ProtobufTranslationMiddleware)
    app.middleware("http")(http_error_translator)
    app.dependency_overrides[get_linkstate] = lambda: state
    return TestClient(app)


def _post(
    client: TestClient, path: str, proto_request: Message, *, token: str | None = None
) -> Response:
    """Send a protobuf request with an optional task token."""
    headers = {"content-type": PROTOBUF_MEDIA_TYPE}
    if token is not None:
        headers[TASK_TOKEN_HEADER] = token
    return cast(
        Response,
        client.post(
            f"/v1/runtime/{path}",
            content=proto_request.SerializeToString(),
            headers=headers,
        ),
    )


def _create_running_task(
    state: LinkState, primary_task_type: str = TaskType.SERVER_APP
) -> str:
    """Create and activate a primary task, returning its token."""
    run_id = state.create_run(
        "", "", "", {}, NOOP_FEDERATION_ID, None, "", primary_task_type
    )
    task_id = state.get_run_info(run_ids=[run_id])[0].primary_task_id
    assert task_id is not None
    token = state.claim_task(task_id)
    assert token is not None
    assert state.activate_task(task_id)
    return token


def test_get_nodes_denied_without_metadata_token(client: TestClient) -> None:
    """Protected routes should deny requests missing a task token."""
    response = _post(client, "get-nodes", GetNodesRequest())

    assert response.status_code == 401
    assert response.json()["code"] == ApiErrorCode.RUNTIME_AUTHENTICATION_FAILED


def test_get_connector_requires_and_uses_connector_task_token(
    client: TestClient, state: LinkState
) -> None:
    """Derive connector credential access from the authenticated task token."""
    assert _post(client, "get-connector", GetConnectorRequest()).status_code == 401
    run_id = state.create_run(
        "",
        "",
        "",
        {},
        NOOP_FEDERATION_ID,
        None,
        "account-a",
        TaskType.AGENT_APP,
        connector_refs=["notion"],
    )
    task_id = state.create_task(TaskType.CONNECTOR, run_id, connector_ref="notion")
    assert task_id is not None
    token = state.claim_task(task_id)
    assert token is not None
    assert state.activate_task(task_id)
    assert state.upsert_connector(
        flwr_aid="account-a",
        connector_ref="notion",
        credentials_json='{"token":"secret"}',
        config_json="{}",
    )

    response = _post(client, "get-connector", GetConnectorRequest(), token=token)

    assert response.status_code == 200
    assert GetConnectorResponse.FromString(response.content) == GetConnectorResponse(
        connector_ref="notion",
        credentials_json='{"token":"secret"}',
        config_json="{}",
    )


def test_get_nodes_denied_with_invalid_metadata_token(client: TestClient) -> None:
    """Protected routes should deny invalid task tokens."""
    response = _post(client, "get-nodes", GetNodesRequest(), token="invalid-token")

    assert response.status_code == 401
    assert response.json()["code"] == ApiErrorCode.RUNTIME_AUTHENTICATION_FAILED


def test_get_nodes_allows_with_valid_metadata_token(
    client: TestClient, state: LinkState
) -> None:
    """Protected routes should allow a valid task token."""
    response = _post(
        client, "get-nodes", GetNodesRequest(), token=_create_running_task(state)
    )

    assert response.status_code == 200
    assert isinstance(GetNodesResponse.FromString(response.content), GetNodesResponse)


@pytest.mark.parametrize(("path", "proto_request"), _SERVERAPP_ONLY_CASES)
def test_serverapp_only_endpoint_denied_for_simulation_run(
    client: TestClient, state: LinkState, path: str, proto_request: Message
) -> None:
    """ServerApp-only routes should deny simulation-task tokens."""
    response = _post(
        client,
        path,
        proto_request,
        token=_create_running_task(state, TaskType.SIMULATION),
    )

    assert response.status_code == 403
    assert response.json()["code"] == ApiErrorCode.RUNTIME_ENDPOINT_UNAVAILABLE


@pytest.mark.parametrize(("path", "proto_request", "response_parser"), _SHARED_CASES)
def test_shared_task_endpoint_allows_simulation_run(
    client: TestClient,
    state: LinkState,
    path: str,
    proto_request: Message,
    response_parser: Callable[[bytes], Message],
) -> None:
    """Shared task routes should allow simulation-task tokens."""
    response = _post(
        client,
        path,
        proto_request,
        token=_create_running_task(state, TaskType.SIMULATION),
    )

    assert response.status_code == 200
    assert response_parser(response.content) is not None
