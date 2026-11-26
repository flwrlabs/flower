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

from typing import cast
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from google.protobuf.message import Message
from httpx import Response

from flwr.proto.control_pb2 import AgentConfig  # pylint: disable=E0611
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    CreateTaskRequest,
    CreateTaskResponse,
    GetConnectorRequest,
    GetConnectorResponse,
    GetNodesRequest,
    GetNodesResponse,
    GetSpaceEventsRequest,
    GetSpaceEventsResponse,
    ListAgentConnectorsRequest,
    ListAgentConnectorsResponse,
)
from flwr.proto.space_pb2 import (  # pylint: disable=E0611
    GetSpaceItemsRequest,
    GetSpaceItemsResponse,
    GetSpaceMessagesRequest,
    GetSpaceMessagesResponse,
    SendSpaceItemRequest,
    SendSpaceItemResponse,
    SendSpaceMessageRequest,
    SendSpaceMessageResponse,
    SpaceParticipant,
)
from flwr.proto.task_pb2 import TaskEvent  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory
from flwr.supercore.constant import (
    FLWR_IN_MEMORY_DB_NAME,
    NOOP_FEDERATION_ID,
    TASK_TOKEN_HEADER,
    TaskType,
)
from flwr.supercore.dependencies.runtime import get_runtime_state
from flwr.supercore.error import ApiErrorCode, http_error_translator
from flwr.supercore.fab import Fab
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.protobuf.constants import PROTOBUF_MEDIA_TYPE
from flwr.supercore.protobuf.translation import ProtobufTranslationMiddleware
from flwr.supercore.routers.runtime import router
from flwr.superlink.federation import NoOpFederationManager
from flwr.superlink.servicer.runtime import runtime_handlers


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
    app.state.runtime_handlers = runtime_handlers
    app.include_router(router)
    app.add_middleware(ProtobufTranslationMiddleware)
    app.middleware("http")(http_error_translator)
    app.dependency_overrides[get_runtime_state] = lambda: state
    return TestClient(app)


@pytest.fixture(name="agent")
def fixture_agent(state: LinkState) -> tuple[int, str, str]:
    """Create the shared Space agent and return its IDs and task token."""
    agent = state.create_agent("alice", AgentConfig(name="Agent", app_spec="@flwr/app"))
    assert agent is not None
    space = state.create_space(participants=[SpaceParticipant(agent_id=agent.id)])
    token = _create_running_task(
        state, TaskType.AGENT_APP, space_id=UUID(space.id), agent_id=agent.id
    )
    return agent.id, space.id, token


def _post(
    client: TestClient, path: str, proto_request: Message, token: str | None = None
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
    state: LinkState,
    primary_task_type: str = TaskType.SERVER_APP,
    *,
    space_id: UUID | None = None,
    agent_id: int | None = None,
) -> str:
    """Create and activate a primary task, returning its token."""
    run_id = state.create_run(
        "flwr/app",
        "1.0",
        state.store_fab(Fab("", b"fab", {})),
        {},
        NOOP_FEDERATION_ID if space_id is None else None,
        None,
        "",
        primary_task_type,
        space_id=space_id,
        agent_id=agent_id,
    )
    assert run_id
    task_id = state.get_run(run_ids=[run_id])[0].primary_task_id
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
    connector_id = state.create_connector(
        federation_id=NOOP_FEDERATION_ID,
        connector_ref="notion",
        credentials_json='{"token":"secret"}',
        config_json="{}",
        created_by="account-a",
    )
    assert connector_id is not None
    run_id = state.create_run(
        "",
        "",
        "",
        {},
        NOOP_FEDERATION_ID,
        None,
        "account-a",
        TaskType.AGENT_APP,
        connector_ids=[connector_id],
    )
    task_id = state.create_task(
        TaskType.CONNECTOR,
        run_id,
        space_id=None,
        connector_ref="notion",
        connector_id=connector_id,
    )
    assert task_id is not None
    token = state.claim_task(task_id)
    assert token is not None
    assert state.activate_task(task_id)
    response = _post(client, "get-connector", GetConnectorRequest(), token=token)

    assert response.status_code == 200
    assert GetConnectorResponse.FromString(response.content) == GetConnectorResponse(
        connector_id=connector_id,
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


def test_get_space_events_reads_only_authenticated_space(
    client: TestClient, state: LinkState
) -> None:
    """A task token only grants access to events in its run's Space."""
    space_id = UUID(state.create_space(participants=[]).id)
    other_space_id = UUID(state.create_space(participants=[]).id)
    run_ids = [
        state.create_run(
            "",
            "",
            "",
            {},
            None,
            None,
            "account",
            TaskType.AGENT_APP,
            space_id=current_space_id,
            initial_task_event=TaskEvent(event="message", data='{"type":"message"}'),
        )
        for current_space_id in (space_id, other_space_id)
    ]
    task_id = state.get_run(run_ids=[run_ids[0]])[0].primary_task_id
    assert task_id is not None
    token = state.claim_task(task_id)
    assert token is not None
    assert state.activate_task(task_id)
    assert state.store_task_events(
        [
            TaskEvent(
                run_id=run_ids[0],
                task_id=task_id,
                event="other",
                data='{"type":"other"}',
            )
        ]
    )

    assert _post(client, "get-space-events", GetSpaceEventsRequest()).status_code == 401
    response = _post(
        client,
        "get-space-events",
        GetSpaceEventsRequest(event_types=["message"], last_n=1),
        token=token,
    )

    assert response.status_code == 200
    events = GetSpaceEventsResponse.FromString(response.content).events
    assert len(events) == 1
    assert events[0].run_id == run_ids[0]
    assert events[0].event == "message"


def test_space_messages_use_authenticated_run(
    client: TestClient, agent: tuple[int, str, str]
) -> None:
    """Read and publish with the authenticated Run's Space and agent identity."""
    agent_id, space_id, token = agent
    participant = SpaceParticipant(agent_id=agent_id)
    for path, request in (
        ("get-space-messages", GetSpaceMessagesRequest()),
        ("send-space-message", SendSpaceMessageRequest(content="Hello")),
    ):
        assert _post(client, path, request).status_code == 401

    response = _post(
        client,
        "send-space-message",
        SendSpaceMessageRequest(space_id="ignored", content="Hello"),
        token=token,
    )
    assert response.status_code == 200
    message = SendSpaceMessageResponse.FromString(response.content).message
    assert message.author == participant
    assert message.space_id == space_id and message.content == "Hello"

    for expected in ([message], []):
        response = _post(
            client,
            "get-space-messages",
            GetSpaceMessagesRequest(space_id="ignored", unread_only=True),
            token=token,
        )
        assert response.status_code == 200
        assert (
            GetSpaceMessagesResponse.FromString(response.content).messages == expected
        )


def test_space_item_endpoints_use_authenticated_run(
    client: TestClient,
    state: LinkState,
    agent: tuple[int, str, str],
) -> None:
    """Read and publish items using the task's Space and agent identity."""
    agent_id, space_id, token = agent
    assert _post(client, "get-space-items", GetSpaceItemsRequest()).status_code == 401
    assert (
        _post(client, "send-space-item", SendSpaceItemRequest(content="Hi")).status_code
        == 401
    )
    items = []
    for content in ("First", "Second"):
        response = _post(
            client,
            "send-space-item",
            SendSpaceItemRequest(space_id="ignored", content=content),
            token=token,
        )
        assert response.status_code == 200
        items.append(SendSpaceItemResponse.FromString(response.content).item)
    assert all(
        item.space_id == space_id
        and item.author == SpaceParticipant(agent_id=agent_id)
        and item.type == "message"
        for item in items
    )
    response = _post(
        client,
        "get-space-items",
        GetSpaceItemsRequest(unread_only=True, limit=1),
        token=token,
    )
    assert response.status_code == 200
    assert list(GetSpaceItemsResponse.FromString(response.content).items) == items[:1]
    response = _post(
        client, "get-space-items", GetSpaceItemsRequest(unread_only=True), token=token
    )
    assert response.status_code == 200
    assert list(GetSpaceItemsResponse.FromString(response.content).items) == items[1:]
    response = _post(
        client, "get-space-items", GetSpaceItemsRequest(unread_only=True), token=token
    )
    assert response.status_code == 200
    assert not GetSpaceItemsResponse.FromString(response.content).items
    assert (
        state.get_space_messages(UUID(space_id), SpaceParticipant(agent_id=agent_id))
        == []
    )
    for path, request in (
        ("get-space-items", GetSpaceItemsRequest(limit=0)),
        ("get-space-items", GetSpaceItemsRequest(limit=101)),
        ("send-space-item", SendSpaceItemRequest(content="  ")),
    ):
        response = _post(client, path, request, token=token)
        assert response.status_code == 400
        assert response.json()["code"] == ApiErrorCode.INVALID_SPACE_REQUEST


def test_agent_connector_discovery_and_credentials(
    client: TestClient, state: LinkState, agent: tuple[int, str, str]
) -> None:
    """List agent connections and use the selected connector task's credentials."""
    agent_id, space_id, token = agent
    connector_id = state.create_connector(
        None, "slack", '{"access_token":"secret"}', "{}", "alice", agent_id
    )
    assert connector_id is not None
    listed = _post(client, "list-agent-connectors", ListAgentConnectorsRequest(), token)
    assert listed.status_code == 200
    connections = ListAgentConnectorsResponse.FromString(listed.content).connectors
    assert [item.connector_id for item in connections] == [connector_id]

    created = _post(
        client,
        "create-task",
        CreateTaskRequest(type=TaskType.CONNECTOR, connector_id=connector_id),
        token,
    )
    assert created.status_code == 200
    child_id = CreateTaskResponse.FromString(created.content).task_id
    assert state.get_tasks(task_ids=[child_id])[0].space_id == space_id
    child_token = state.claim_task(child_id)
    assert child_token is not None and state.activate_task(child_id)
    response = _post(client, "get-connector", GetConnectorRequest(), child_token)
    assert response.status_code == 200
    assert (
        GetConnectorResponse.FromString(response.content).credentials_json
        == '{"access_token":"secret"}'
    )


def test_agent_connector_rejects_foreign_connection(
    client: TestClient, state: LinkState, agent: tuple[int, str, str]
) -> None:
    """Hide another agent's connection and deny its use."""
    _, _, token = agent
    other = state.create_agent("bob", AgentConfig(name="Other", app_spec="@test/app"))
    assert other is not None
    foreign = state.create_connector(None, "slack", "{}", "{}", "bob", other.id)
    assert foreign is not None
    listed = _post(client, "list-agent-connectors", ListAgentConnectorsRequest(), token)
    assert listed.status_code == 200
    assert not ListAgentConnectorsResponse.FromString(listed.content).connectors
    response = _post(
        client,
        "create-task",
        CreateTaskRequest(type=TaskType.CONNECTOR, connector_id=foreign),
        token,
    )
    assert response.status_code == 403
