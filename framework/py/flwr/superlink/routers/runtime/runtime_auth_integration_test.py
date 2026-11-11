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
    GetConnectorRequest,
    GetConnectorResponse,
    GetNodesRequest,
    GetNodesResponse,
    GetSpaceEventsRequest,
    GetSpaceEventsResponse,
)
from flwr.proto.space_pb2 import (  # pylint: disable=E0611
    GetSpaceMessagesRequest,
    GetSpaceMessagesResponse,
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
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.protobuf.constants import PROTOBUF_MEDIA_TYPE
from flwr.supercore.protobuf.translation import ProtobufTranslationMiddleware
from flwr.supercore.routers.runtime import router
from flwr.superlink.federation import NoOpFederationManager
from flwr.superlink.servicer.runtime import runtime_handlers


@pytest.fixture(name="state", params=[FLWR_IN_MEMORY_DB_NAME, ":memory:"])
def fixture_state(request: pytest.FixtureRequest) -> LinkState:
    """Create in-memory and SQLite LinkState backends."""
    return LinkStateFactory(
        request.param,
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
    state: LinkState,
    primary_task_type: str = TaskType.SERVER_APP,
    *,
    space_id: UUID | None = None,
    agent_id: int | None = None,
) -> str:
    """Create and activate a primary task, returning its token."""
    run_id = state.create_run(
        "",
        "",
        "",
        {},
        NOOP_FEDERATION_ID if space_id is None else None,
        None,
        "",
        primary_task_type,
        space_id=space_id,
        agent_id=agent_id,
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
    task_id = state.get_run_info(run_ids=[run_ids[0]])[0].primary_task_id
    assert task_id is not None
    token = state.claim_task(task_id)
    assert token is not None
    assert state.activate_task(task_id)

    assert _post(client, "get-space-events", GetSpaceEventsRequest()).status_code == 401
    response = _post(client, "get-space-events", GetSpaceEventsRequest(), token=token)

    assert response.status_code == 200
    events = GetSpaceEventsResponse.FromString(response.content).events
    assert len(events) == 1
    assert events[0].run_id == run_ids[0]


def test_space_messages_use_authenticated_run(
    client: TestClient, state: LinkState
) -> None:
    """Read and publish with the authenticated Run's Space and agent identity."""
    agents = []
    for name in ("first", "second"):
        agent = state.create_agent(
            "alice", AgentConfig(name=name, app_spec="@flwr/app")
        )
        assert agent is not None
        agents.append(agent)
    author = SpaceParticipant(flwr_aid="alice")
    space = state.create_space(
        participants=[author, *(SpaceParticipant(agent_id=a.id) for a in agents)]
    )
    other = state.create_space(participants=[author])
    token = _create_running_task(
        state, TaskType.AGENT_APP, space_id=UUID(space.id), agent_id=agents[0].id
    )
    assert (
        _post(client, "get-space-messages", GetSpaceMessagesRequest()).status_code
        == 401
    )
    assert (
        _post(client, "send-space-message", SendSpaceMessageRequest()).status_code
        == 401
    )
    response = _post(
        client, "get-space-messages", GetSpaceMessagesRequest(), token=token
    )
    assert response.status_code == 200
    assert not GetSpaceMessagesResponse.FromString(response.content).messages

    first = state.create_space_message(UUID(space.id), author, "Hello")
    assert first is not None
    assert state.create_space_message(UUID(other.id), author, "Private") is not None
    response = _post(
        client,
        "send-space-message",
        SendSpaceMessageRequest(content="Reply\nas agent"),
        token=token,
    )
    assert response.status_code == 200
    reply = SendSpaceMessageResponse.FromString(response.content).message
    assert reply.author == SpaceParticipant(agent_id=agents[0].id)
    assert reply.space_id == space.id and reply.content == "Reply\nas agent"

    request: Message
    for request, expected in (
        (GetSpaceMessagesRequest(unread_only=True, limit=1), [first]),
        (GetSpaceMessagesRequest(unread_only=True, limit=1), [reply]),
        (GetSpaceMessagesRequest(unread_only=True), []),
        (GetSpaceMessagesRequest(), [first, reply]),
        (GetSpaceMessagesRequest(after_id=0, limit=1), [first]),
        (GetSpaceMessagesRequest(before_id=reply.id), [first]),
        (GetSpaceMessagesRequest(after_id=first.id), [reply]),
    ):
        response = _post(client, "get-space-messages", request, token=token)
        assert response.status_code == 200
        assert (
            GetSpaceMessagesResponse.FromString(response.content).messages == expected
        )

    for path, request in (
        ("get-space-messages", GetSpaceMessagesRequest(space_id=other.id)),
        (
            "send-space-message",
            SendSpaceMessageRequest(space_id=other.id, content="No"),
        ),
    ):
        response = _post(client, path, request, token=token)
        assert response.status_code == 404
        assert response.json()["code"] == ApiErrorCode.SPACE_NOT_FOUND

    for path, request in (
        ("get-space-messages", GetSpaceMessagesRequest(limit=0)),
        ("get-space-messages", GetSpaceMessagesRequest(limit=101)),
        ("get-space-messages", GetSpaceMessagesRequest(space_id="invalid")),
        ("get-space-messages", GetSpaceMessagesRequest(after_id=2, before_id=1)),
        ("send-space-message", SendSpaceMessageRequest(content=" \n")),
    ):
        response = _post(client, path, request, token=token)
        assert response.status_code == 400
        assert response.json()["code"] == ApiErrorCode.INVALID_SPACE_REQUEST


@pytest.mark.parametrize(
    "path,proto_request",
    [
        ("get-space-messages", GetSpaceMessagesRequest()),
        ("send-space-message", SendSpaceMessageRequest(content="Hello")),
    ],
)
def test_space_messages_reject_legacy_tasks(
    client: TestClient, state: LinkState, path: str, proto_request: Message
) -> None:
    """A legacy Run does not grant access to Space messages."""
    response = _post(client, path, proto_request, token=_create_running_task(state))
    assert response.status_code == 403
    assert response.json()["code"] == ApiErrorCode.RUNTIME_ENDPOINT_UNAVAILABLE


@pytest.mark.parametrize(
    "path,proto_request",
    [
        ("get-space-messages", GetSpaceMessagesRequest()),
        ("send-space-message", SendSpaceMessageRequest(content="Hello")),
    ],
)
def test_space_messages_reject_invalid_task_tokens(
    client: TestClient, path: str, proto_request: Message
) -> None:
    """Reject invalid task tokens before resolving Space access."""
    response = _post(client, path, proto_request, token="invalid-token")
    assert response.status_code == 401
    assert response.json()["code"] == ApiErrorCode.RUNTIME_AUTHENTICATION_FAILED
