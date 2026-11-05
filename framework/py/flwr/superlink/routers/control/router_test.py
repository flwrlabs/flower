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
"""Tests for the Control API router."""

# pylint: disable=too-many-lines

import inspect
from collections.abc import Callable
from datetime import datetime
from unittest.mock import Mock, call, patch
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from google.protobuf.json_format import ParseDict
from google.protobuf.message import Message

from flwr.common.constant import (
    ACCESS_TOKEN_KEY,
    NOOP_FLWR_AID,
    REFRESH_TOKEN_KEY,
    Status,
    SubStatus,
)
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    AddAppRequest,
    AddAppResponse,
    AgentConfig,
    BeginConnectorOAuthRequest,
    BeginConnectorOAuthResponse,
    CompleteConnectorOAuthRequest,
    CompleteConnectorOAuthResponse,
    Connector,
    CreateAgentRequest,
    CreateAgentResponse,
    CreateSpaceRequest,
    CreateSpaceResponse,
    DisconnectConnectorRequest,
    DisconnectConnectorResponse,
    GetAuthTokensRequest,
    GetAuthTokensResponse,
    GetLoginDetailsRequest,
    GetLoginDetailsResponse,
    GetSpaceMessagesRequest,
    GetSpaceMessagesResponse,
    ListAgentsRequest,
    ListAgentsResponse,
    ListConnectorsRequest,
    ListConnectorsResponse,
    ListRunsRequest,
    ListRunsResponse,
    ListSpacesRequest,
    ListSpacesResponse,
    PullArtifactsRequest,
    PullArtifactsResponse,
    RefreshAuthTokensRequest,
    RefreshAuthTokensResponse,
    SendSpaceMessageRequest,
    SendSpaceMessageResponse,
    SpaceParticipant,
    StartRunRequest,
    StartRunResponse,
    StreamLogsRequest,
    StreamLogsResponse,
    StreamRunEventsRequest,
    StreamRunEventsResponse,
)
from flwr.proto.task_pb2 import TaskEvent  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory
from flwr.supercore.auth.typing import (
    AccountAuthCredentials,
    AccountAuthLoginDetails,
    AccountInfo,
)
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME
from flwr.supercore.error import ApiErrorCode, http_error_translator
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.protobuf.constants import (
    PROTOBUF_MEDIA_TYPE,
    PROTOBUF_STREAM_MEDIA_TYPE,
)
from flwr.supercore.protobuf.framing import frame_message
from flwr.supercore.protobuf.translation import (
    PROTOBUF_REQUEST_TYPES,
    ProtobufTranslationMiddleware,
    get_protobuf_request,
)
from flwr.supercore.run import Run, RunStatus
from flwr.superlink.artifact_provider import ArtifactProvider
from flwr.superlink.dependencies.account import AccountAccessDependency
from flwr.superlink.dependencies.linkstate import get_linkstate
from flwr.superlink.federation import NoOpFederationManager
from flwr.superlink.routers.control.middlewares import ControlAuthenticationMiddleware
from flwr.superlink.routers.control.router import add_app as add_app_route
from flwr.superlink.routers.control.router import router
from flwr.superlink.routers.control.router import start_run as start_run_route
from flwr.superlink.servicer.control import control_handlers

_ACCOUNT = AccountInfo(flwr_aid=NOOP_FLWR_AID, account_name="account")
CONNECTOR_FEDERATION_ID = "@bob/fed-a"


@pytest.fixture(name="space_client")
def space_client_fixture() -> tuple[TestClient, LinkState]:
    """Provide authenticated space routes backed by in-memory state."""
    state = LinkStateFactory(
        FLWR_IN_MEMORY_DB_NAME, NoOpFederationManager(), ObjectStoreFactory()
    ).state()
    app = _create_app()
    app.dependency_overrides[get_linkstate] = lambda: state
    return (
        TestClient(
            app,
            headers={
                "authorization": "Bearer access-token",
                "content-type": PROTOBUF_MEDIA_TYPE,
            },
        ),
        state,
    )


def test_space_routes(
    space_client: tuple[TestClient, LinkState],
) -> None:
    """Create, list, send, and read with authenticated caller membership."""
    client, state = space_client
    agent = state.create_agent(
        flwr_aid=_ACCOUNT.flwr_aid,
        config=AgentConfig(name="Research", app_spec="@flower/research"),
    )
    assert agent is not None
    caller = SpaceParticipant(flwr_aid=_ACCOUNT.flwr_aid)
    agent_participant = SpaceParticipant(agent_id=agent.id)
    response = client.post(
        "/v1/control/create-space",
        content=CreateSpaceRequest(
            participants=[agent_participant]
        ).SerializeToString(),
    )
    assert response.status_code == 200
    space = CreateSpaceResponse.FromString(response.content).space
    assert list(space.participants) == [caller, agent_participant]
    listed = client.post(
        "/v1/control/list-spaces",
        content=ListSpacesRequest().SerializeToString(),
    )
    assert listed.status_code == 200
    assert list(ListSpacesResponse.FromString(listed.content).spaces) == [space]
    response = client.post(
        "/v1/control/send-space-message",
        content=SendSpaceMessageRequest(
            space_id=space.id, content="Hello"
        ).SerializeToString(),
    )
    assert response.status_code == 200
    message = SendSpaceMessageResponse.FromString(response.content).message
    assert message.author == caller and message.content == "Hello"
    history = client.post(
        "/v1/control/get-space-messages",
        content=GetSpaceMessagesRequest(space_id=space.id).SerializeToString(),
    )
    assert history.status_code == 200
    assert list(GetSpaceMessagesResponse.FromString(history.content).messages) == [
        message
    ]
    assert not state.get_run_series() and not state.get_tasks()


def test_create_space_rejects_invalid_participants(
    space_client: tuple[TestClient, LinkState],
) -> None:
    """Reject invalid initial participants without creating a space."""
    client, state = space_client
    response = client.post(
        "/v1/control/create-space",
        content=CreateSpaceRequest(
            participants=[SpaceParticipant()]
        ).SerializeToString(),
    )
    assert response.status_code == 400
    assert response.json()["code"] == ApiErrorCode.INVALID_SPACE_REQUEST
    assert not state.get_spaces(_ACCOUNT.flwr_aid)


def test_create_space_rejects_other_users_agent(
    space_client: tuple[TestClient, LinkState],
) -> None:
    """An account cannot attach another account's private agent."""
    client, state = space_client
    agent = state.create_agent(
        flwr_aid="other-user",
        config=AgentConfig(name="Research", app_spec="@flower/research"),
    )
    assert agent is not None
    response = client.post(
        "/v1/control/create-space",
        content=CreateSpaceRequest(
            participants=[SpaceParticipant(agent_id=agent.id)]
        ).SerializeToString(),
    )
    assert response.status_code == 400
    assert response.json()["code"] == ApiErrorCode.INVALID_SPACE_REQUEST
    assert not state.get_spaces(_ACCOUNT.flwr_aid)


def test_space_history_pagination(
    space_client: tuple[TestClient, LinkState],
) -> None:
    """Apply the HTTP request's exclusive cursor and page size."""
    client, state = space_client
    space = state.create_space(
        participants=[SpaceParticipant(flwr_aid=_ACCOUNT.flwr_aid)]
    )
    space_id = UUID(space.id)
    author = SpaceParticipant(flwr_aid=_ACCOUNT.flwr_aid)
    messages = [state.create_space_message(space_id, author, str(i)) for i in range(3)]
    last = messages[-1]
    assert last is not None
    response = client.post(
        "/v1/control/get-space-messages",
        content=GetSpaceMessagesRequest(
            space_id=space.id, before_id=last.id, limit=1
        ).SerializeToString(),
    )
    assert response.status_code == 200
    assert list(GetSpaceMessagesResponse.FromString(response.content).messages) == [
        messages[1]
    ]


@pytest.mark.parametrize("endpoint", ["get-space-messages", "send-space-message"])
def test_space_unavailable_to_nonparticipants(
    space_client: tuple[TestClient, LinkState], endpoint: str
) -> None:
    """Missing and inaccessible spaces produce the same 404 response."""
    client, state = space_client
    private = state.create_space(participants=[SpaceParticipant(flwr_aid="alice")])
    for space_id in [private.id, str(uuid4())]:
        request = (
            GetSpaceMessagesRequest(space_id=space_id)
            if endpoint == "get-space-messages"
            else SendSpaceMessageRequest(space_id=space_id, content="No access")
        )
        response = client.post(
            f"/v1/control/{endpoint}", content=request.SerializeToString()
        )
        assert response.status_code == 404
        assert response.json()["code"] == ApiErrorCode.SPACE_NOT_FOUND


@pytest.mark.parametrize(
    "endpoint,fields",
    [
        ("send-space-message", {"content": " \n"}),
        (
            "send-space-message",
            {"content": "Hello", "space_id": "bad-uuid"},
        ),
        ("get-space-messages", {"space_id": "bad-uuid"}),
        ("get-space-messages", {"limit": 0}),
        ("get-space-messages", {"limit": 101}),
    ],
)
def test_invalid_space_request(
    space_client: tuple[TestClient, LinkState],
    endpoint: str,
    fields: dict[str, str | int],
) -> None:
    """Reject malformed IDs, empty text, and unsupported page sizes."""
    client, _ = space_client
    request_type = (
        SendSpaceMessageRequest
        if endpoint == "send-space-message"
        else GetSpaceMessagesRequest
    )
    request = ParseDict({"space_id": str(uuid4()), **fields}, request_type())
    response = client.post(
        f"/v1/control/{endpoint}", content=request.SerializeToString()
    )
    assert response.status_code == 400
    assert response.json()["code"] == ApiErrorCode.INVALID_SPACE_REQUEST


def _create_app(authn_plugin: Mock | None = None) -> FastAPI:
    """Create a minimal app containing the Control API stack."""
    authn_plugin = authn_plugin or Mock()
    authn_plugin.validate_tokens_in_metadata.return_value = (True, _ACCOUNT)
    app = FastAPI()
    app.state.account_access_dep = AccountAccessDependency(authn_plugin)
    app.include_router(router)
    app.add_middleware(ProtobufTranslationMiddleware)
    app.add_middleware(ControlAuthenticationMiddleware)
    app.middleware("http")(http_error_translator)
    return app


def test_all_control_routes_have_protobuf_request_types() -> None:
    """Every Control route has exactly one protobuf request type mapping."""
    route_keys = {
        (method, route.path)
        for route in router.routes
        if isinstance(route, APIRoute)
        for method in (route.methods or set())
    }

    control_request_types = {
        route_key
        for route_key in PROTOBUF_REQUEST_TYPES
        if route_key[1].startswith("/v1/control/")
    }
    assert route_keys == control_request_types


def test_all_control_routes_declare_protobuf_openapi_contracts() -> None:
    """Document every Control route as protobuf-over-HTTP."""
    schema = _create_app().openapi()
    control_routes = [route for route in router.routes if isinstance(route, APIRoute)]
    stream_paths = {
        "/v1/control/stream-logs",
        "/v1/control/stream-run-events",
    }

    assert control_routes
    for route in control_routes:
        operation = schema["paths"][route.path]["post"]
        request_body = operation["requestBody"]
        success_content = operation["responses"]["200"]["content"]
        expected_media_type = (
            PROTOBUF_STREAM_MEDIA_TYPE
            if route.path in stream_paths
            else PROTOBUF_MEDIA_TYPE
        )

        assert operation["description"] == inspect.getdoc(
            inspect.unwrap(route.endpoint)
        )
        assert request_body["required"] is True
        assert request_body["content"] == {
            PROTOBUF_MEDIA_TYPE: {"schema": {"type": "string", "format": "binary"}}
        }
        assert set(success_content) == {expected_media_type}
        assert success_content[expected_media_type]["schema"] == {
            "type": "string",
            "format": "binary",
        }


@pytest.mark.parametrize("database", [FLWR_IN_MEMORY_DB_NAME, ":memory:"])
def test_agent_routes_create_and_list_owned_agents(database: str) -> None:
    """Persist an agent through HTTP and list only the authenticated owner's agents."""
    state = LinkStateFactory(
        database, NoOpFederationManager(), ObjectStoreFactory()
    ).state()
    config = AgentConfig(name="Research Agent", app_spec="@flower/research==1.2.3")
    assert state.create_agent(flwr_aid="other-account", config=config) is not None
    app = _create_app()
    app.dependency_overrides[get_linkstate] = lambda: state
    client = TestClient(app)
    headers = {
        "authorization": "Bearer access-token",
        "content-type": PROTOBUF_MEDIA_TYPE,
    }

    response = client.post(
        "/v1/control/create-agent",
        content=CreateAgentRequest(config=config).SerializeToString(),
        headers=headers,
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == PROTOBUF_MEDIA_TYPE
    agent = CreateAgentResponse.FromString(response.content).agent
    assert agent.id > 0
    assert agent.config == config
    assert agent.created_at and agent.updated_at == agent.created_at
    assert state.list_agents(_ACCOUNT.flwr_aid) == [agent]

    response = client.post(
        "/v1/control/list-agents",
        content=ListAgentsRequest().SerializeToString(),
        headers=headers,
    )
    assert response.status_code == 200
    assert ListAgentsResponse.FromString(response.content).agents == [agent]


@pytest.mark.parametrize(
    ("protobuf_request", "detail"),
    [
        (CreateAgentRequest(), "configuration"),
        (
            CreateAgentRequest(config=AgentConfig(app_spec="@flower/research")),
            "name",
        ),
        (
            CreateAgentRequest(
                config=AgentConfig(name="  ", app_spec="@flower/research")
            ),
            "name",
        ),
        (CreateAgentRequest(config=AgentConfig(name="Research")), "app"),
        (
            CreateAgentRequest(config=AgentConfig(name="Research", app_spec="invalid")),
            "app",
        ),
        (
            CreateAgentRequest(
                config=AgentConfig(
                    name="Research", app_spec="@flower/research", title=""
                )
            ),
            "title",
        ),
        (
            CreateAgentRequest(
                config=AgentConfig(
                    name="Research", app_spec="@flower/research", description=""
                )
            ),
            "description",
        ),
    ],
)
def test_agent_creation_rejects_invalid_requests(
    protobuf_request: CreateAgentRequest, detail: str
) -> None:
    """Report invalid input as HTTP 400 without attempting to store an agent."""
    state = Mock(spec=LinkState)
    app = _create_app()
    app.dependency_overrides[get_linkstate] = lambda: state

    response = TestClient(app).post(
        "/v1/control/create-agent",
        content=protobuf_request.SerializeToString(),
        headers={
            "authorization": "Bearer access-token",
            "content-type": PROTOBUF_MEDIA_TYPE,
        },
    )

    assert response.status_code == 400
    assert response.json()["code"] == ApiErrorCode.INVALID_AGENT_REQUEST.value
    assert detail in response.json()["extra"].lower()
    state.create_agent.assert_not_called()


@pytest.mark.parametrize(
    ("path", "protobuf_request", "parse_response", "expected", "handler_name"),
    [
        (
            "/v1/control/list-connectors",
            ListConnectorsRequest(federation=CONNECTOR_FEDERATION_ID),
            ListConnectorsResponse.FromString,
            ListConnectorsResponse(
                connectors=[
                    Connector(
                        connector_ref="google-drive",
                        display_name="Google Drive",
                        description="Cloud storage",
                        connected=True,
                    )
                ]
            ),
            "list_connectors",
        ),
        (
            "/v1/control/disconnect-connector",
            DisconnectConnectorRequest(
                connector_ref="google-drive", federation=CONNECTOR_FEDERATION_ID
            ),
            DisconnectConnectorResponse.FromString,
            DisconnectConnectorResponse(),
            "disconnect_connector",
        ),
        (
            "/v1/control/begin-connector-oauth",
            BeginConnectorOAuthRequest(
                connector_ref="google-drive",
                redirect_uri="https://example.test/oauth/callback",
                federation=CONNECTOR_FEDERATION_ID,
            ),
            BeginConnectorOAuthResponse.FromString,
            BeginConnectorOAuthResponse(
                oauth_session_id="oauth-session",
                authorization_url="https://provider.test/authorize",
                connector_ref="google-drive",
                expires_at="2026-08-27T12:00:00+00:00",
            ),
            "begin_connector_oauth",
        ),
        (
            "/v1/control/complete-connector-oauth",
            CompleteConnectorOAuthRequest(
                oauth_session_id="oauth-session",
                code="authorization-code",
                state="oauth-state",
            ),
            CompleteConnectorOAuthResponse.FromString,
            CompleteConnectorOAuthResponse(connector_ref="google-drive"),
            "complete_connector_oauth",
        ),
    ],
)
def test_connector_routes_return_protobuf_responses(
    path: str,
    protobuf_request: Message,
    parse_response: Callable[[bytes], Message],
    expected: Message,
    handler_name: str,
) -> None:
    """Forward authenticated connector requests and serialize their responses."""
    linkstate = Mock(spec=LinkState)
    app = _create_app()
    app.dependency_overrides[get_linkstate] = lambda: linkstate

    with patch.object(
        control_handlers,
        handler_name,
        return_value=expected,
    ) as handler:
        response = TestClient(app).post(
            path,
            content=protobuf_request.SerializeToString(),
            headers={
                "authorization": "Bearer access-token",
                "content-type": PROTOBUF_MEDIA_TYPE,
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == PROTOBUF_MEDIA_TYPE
    assert parse_response(response.content) == expected
    handler.assert_called_once_with(protobuf_request, _ACCOUNT, linkstate)


def test_connector_route_returns_existing_structured_error() -> None:
    """Translate connector handler errors without changing their public contract."""
    linkstate = Mock(spec=LinkState)
    app = _create_app()
    app.dependency_overrides[get_linkstate] = lambda: linkstate

    response = TestClient(app).post(
        "/v1/control/begin-connector-oauth",
        content=BeginConnectorOAuthRequest(
            federation=CONNECTOR_FEDERATION_ID
        ).SerializeToString(),
        headers={
            "authorization": "Bearer access-token",
            "content-type": PROTOBUF_MEDIA_TYPE,
        },
    )

    assert response.status_code == 400
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {
        "detail": "Invalid connector request.",
        "code": ApiErrorCode.INVALID_CONNECTOR_REQUEST.value,
    }


@pytest.mark.parametrize(
    ("authorization", "expected_metadata"),
    [
        (None, []),
        ("Bearer invalid-token", [(ACCESS_TOKEN_KEY, "invalid-token")]),
    ],
)
@pytest.mark.parametrize(
    ("path", "protobuf_request", "handler_name"),
    [
        ("/v1/control/create-agent", CreateAgentRequest(), "create_agent"),
        ("/v1/control/list-agents", ListAgentsRequest(), "list_agents"),
        (
            "/v1/control/list-connectors",
            ListConnectorsRequest(),
            "list_connectors",
        ),
        (
            "/v1/control/disconnect-connector",
            DisconnectConnectorRequest(),
            "disconnect_connector",
        ),
        (
            "/v1/control/begin-connector-oauth",
            BeginConnectorOAuthRequest(),
            "begin_connector_oauth",
        ),
        (
            "/v1/control/complete-connector-oauth",
            CompleteConnectorOAuthRequest(),
            "complete_connector_oauth",
        ),
    ],
)
def test_control_routes_reject_invalid_access_tokens_without_refresh(
    path: str,
    protobuf_request: Message,
    handler_name: str,
    authorization: str | None,
    expected_metadata: list[tuple[str, str]],
) -> None:
    """Require an access token without attempting a refresh-token flow."""
    authn_plugin = Mock()
    app = _create_app(authn_plugin)
    authn_plugin.validate_tokens_in_metadata.return_value = (False, None)
    app.dependency_overrides[get_linkstate] = lambda: Mock(spec=LinkState)
    headers = {"content-type": PROTOBUF_MEDIA_TYPE}
    if authorization is not None:
        headers["authorization"] = authorization

    with patch.object(control_handlers, handler_name) as handler:
        response = TestClient(app).post(
            path,
            content=protobuf_request.SerializeToString(),
            headers=headers,
        )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json() == {"detail": "Not authenticated"}
    authn_plugin.validate_tokens_in_metadata.assert_called_once_with(expected_metadata)
    authn_plugin.refresh_tokens.assert_not_called()
    handler.assert_not_called()


def test_get_login_details_does_not_require_bearer_authentication() -> None:
    """Return login details without validating an access token."""
    authn_plugin = Mock()
    authn_plugin.validate_tokens_in_metadata.return_value = (False, None)
    authn_plugin.get_login_details.return_value = AccountAuthLoginDetails(
        authn_type="oidc",
        device_code="device-code",
        verification_uri_complete="https://example.test/verify",
        expires_in=600,
        interval=5,
    )

    response = TestClient(_create_app(authn_plugin)).post(
        "/v1/control/get-login-details",
        content=GetLoginDetailsRequest().SerializeToString(),
        headers={"content-type": PROTOBUF_MEDIA_TYPE},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == PROTOBUF_MEDIA_TYPE
    assert GetLoginDetailsResponse.FromString(response.content) == (
        GetLoginDetailsResponse(
            authn_type="oidc",
            device_code="device-code",
            verification_uri_complete="https://example.test/verify",
            expires_in=600,
            interval=5,
        )
    )
    authn_plugin.validate_tokens_in_metadata.assert_not_called()


def test_get_auth_tokens_preserves_polling_semantics() -> None:
    """Return an empty response while pending and tokens after authorization."""
    authn_plugin = Mock()
    authn_plugin.validate_tokens_in_metadata.return_value = (False, None)
    authn_plugin.get_auth_tokens.side_effect = [
        None,
        AccountAuthCredentials(
            access_token="access-token",
            refresh_token="refresh-token",
        ),
    ]
    client = TestClient(_create_app(authn_plugin))
    request = GetAuthTokensRequest(device_code="device-code")

    pending_response = client.post(
        "/v1/control/get-auth-tokens",
        content=request.SerializeToString(),
        headers={"content-type": PROTOBUF_MEDIA_TYPE},
    )
    completed_response = client.post(
        "/v1/control/get-auth-tokens",
        content=request.SerializeToString(),
        headers={"content-type": PROTOBUF_MEDIA_TYPE},
    )

    assert pending_response.status_code == 200
    assert pending_response.headers["content-type"] == PROTOBUF_MEDIA_TYPE
    assert GetAuthTokensResponse.FromString(pending_response.content) == (
        GetAuthTokensResponse()
    )
    assert completed_response.status_code == 200
    assert completed_response.headers["content-type"] == PROTOBUF_MEDIA_TYPE
    assert GetAuthTokensResponse.FromString(completed_response.content) == (
        GetAuthTokensResponse(
            access_token="access-token",
            refresh_token="refresh-token",
        )
    )
    authn_plugin.validate_tokens_in_metadata.assert_not_called()


def test_refresh_auth_tokens_does_not_require_bearer_authentication() -> None:
    """Exchange a refresh token without first validating an access token."""
    authn_plugin = Mock()
    authn_plugin.validate_tokens_in_metadata.return_value = (False, None)
    authn_plugin.refresh_tokens.return_value = (
        [
            (ACCESS_TOKEN_KEY, "new-access-token"),
            (REFRESH_TOKEN_KEY, "new-refresh-token"),
        ],
        _ACCOUNT,
    )

    response = TestClient(_create_app(authn_plugin)).post(
        "/v1/control/refresh-auth-tokens",
        content=RefreshAuthTokensRequest(
            refresh_token="old-refresh-token"
        ).SerializeToString(),
        headers={"content-type": PROTOBUF_MEDIA_TYPE},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == PROTOBUF_MEDIA_TYPE
    assert RefreshAuthTokensResponse.FromString(response.content) == (
        RefreshAuthTokensResponse(
            access_token="new-access-token",
            refresh_token="new-refresh-token",
        )
    )
    authn_plugin.validate_tokens_in_metadata.assert_not_called()
    authn_plugin.refresh_tokens.assert_called_once_with(
        [(REFRESH_TOKEN_KEY, "old-refresh-token")]
    )


def test_refresh_auth_tokens_returns_sanitized_authentication_error() -> None:
    """Return the established JSON error without exposing the refresh token."""
    authn_plugin = Mock()
    authn_plugin.refresh_tokens.return_value = (None, None)

    response = TestClient(_create_app(authn_plugin)).post(
        "/v1/control/refresh-auth-tokens",
        content=RefreshAuthTokensRequest(
            refresh_token="secret-refresh-token"
        ).SerializeToString(),
        headers={"content-type": PROTOBUF_MEDIA_TYPE},
    )

    assert response.status_code == 401
    assert response.headers["content-type"] == "application/json"
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json() == {
        "detail": "Authentication failed.",
        "code": ApiErrorCode.ACCOUNT_AUTHENTICATION_FAILED.value,
    }
    assert b"secret-refresh-token" not in response.content


def test_start_run_forwards_resolved_source() -> None:
    """Forward the normalized source to the control handler."""
    request = StartRunRequest()
    linkstate = Mock()
    expected = StartRunResponse(run_id=1)

    with patch(
        "flwr.superlink.routers.control.router.control_handlers.start_run",
        return_value=expected,
    ) as start_run:
        response = start_run_route(
            request,
            linkstate,
            _ACCOUNT,
            "unknown",
            Mock(),
        )

    assert response is expected
    start_run.assert_called_once_with(
        request,
        _ACCOUNT,
        linkstate,
        source="unknown",
    )


def test_start_run_forwards_caller_provided_source() -> None:
    """Forward a caller-provided analytics source without treating it as auth."""
    request = StartRunRequest()
    linkstate = Mock()
    expected = StartRunResponse(run_id=1)

    with patch(
        "flwr.superlink.routers.control.router.control_handlers.start_run",
        return_value=expected,
    ) as start_run:
        start_run_route(
            request,
            linkstate,
            _ACCOUNT,
            run_source="cli",
            tracing=Mock(),
        )

    assert start_run.call_args.kwargs["source"] == "cli"


def test_start_run_reads_client_metadata_header() -> None:
    """Read and normalize the client metadata header through the HTTP stack."""
    linkstate = Mock(spec=LinkState)
    app = _create_app()
    app.dependency_overrides[get_linkstate] = lambda: linkstate
    expected = StartRunResponse(run_id=1)

    with patch.object(
        control_handlers,
        "start_run",
        return_value=expected,
    ) as start_run:
        response = TestClient(app).post(
            "/v1/control/start-run",
            content=StartRunRequest().SerializeToString(),
            headers={
                "authorization": "Bearer access-token",
                "content-type": PROTOBUF_MEDIA_TYPE,
                "x-flwr-client": "web_ui",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == PROTOBUF_MEDIA_TYPE
    assert StartRunResponse.FromString(response.content) == expected
    start_run.assert_called_once_with(
        StartRunRequest(),
        _ACCOUNT,
        linkstate,
        source="web_ui",
    )


def test_add_app_forwards_request() -> None:
    """Forward the request to the control handler."""
    request = AddAppRequest()
    linkstate = Mock()
    expected = AddAppResponse()

    with patch(
        "flwr.superlink.routers.control.router.control_handlers.add_app",
        return_value=expected,
    ) as add_app:
        response = add_app_route(request, linkstate, _ACCOUNT)

    assert response is expected
    add_app.assert_called_once_with(request, _ACCOUNT, linkstate)


def test_protobuf_request_without_handler_response_returns_internal_error() -> None:
    """A configured protobuf route must store its handler response in request state."""
    app = FastAPI()

    # See this route doesn't return a protobuf object
    @app.post("/v1/control/list-runs")
    def list_runs() -> Response:
        return Response()

    app.add_middleware(ProtobufTranslationMiddleware)
    app.middleware("http")(http_error_translator)

    response = TestClient(app).post(
        "/v1/control/list-runs",
        content=ListRunsRequest().SerializeToString(),
        headers={"content-type": PROTOBUF_MEDIA_TYPE},
    )

    assert response.status_code == 500
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {
        "detail": "Invalid protobuf response.",
        "code": ApiErrorCode.INVALID_PROTOBUF_RESPONSE.value,
    }


def test_protobuf_route_passes_through_http_exception() -> None:
    """Return completed JSON errors without requiring a protobuf response."""
    app = FastAPI()

    @app.post("/v1/control/list-runs")
    def list_runs() -> None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Run already exists.",
        )

    app.add_middleware(ProtobufTranslationMiddleware)
    app.middleware("http")(http_error_translator)

    response = TestClient(app).post(
        "/v1/control/list-runs",
        content=ListRunsRequest().SerializeToString(),
        headers={
            "accept": PROTOBUF_MEDIA_TYPE,
            "content-type": PROTOBUF_MEDIA_TYPE,
        },
    )

    assert response.status_code == 409
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"detail": "Run already exists."}


def test_protobuf_route_passes_through_validation_error() -> None:
    """Return FastAPI request validation errors as JSON."""
    app = FastAPI()

    @app.post("/v1/control/list-runs")
    def list_runs(limit: int) -> None:  # pylint: disable=unused-argument
        return None

    app.add_middleware(ProtobufTranslationMiddleware)
    app.middleware("http")(http_error_translator)

    response = TestClient(app).post(
        "/v1/control/list-runs?limit=invalid",
        content=ListRunsRequest().SerializeToString(),
        headers={"content-type": PROTOBUF_MEDIA_TYPE},
    )

    assert response.status_code == 422
    assert response.headers["content-type"] == "application/json"
    payload = response.json()
    assert set(payload) == {"detail"}
    assert isinstance(payload["detail"], list)


def test_protobuf_route_rejects_non_json_error_response() -> None:
    """Reject non-JSON errors instead of bypassing protobuf validation."""
    app = FastAPI()

    @app.post("/v1/control/list-runs")
    def list_runs() -> Response:
        return Response(
            content="Conflict.",
            status_code=status.HTTP_409_CONFLICT,
            media_type="text/plain",
        )

    app.add_middleware(ProtobufTranslationMiddleware)
    app.middleware("http")(http_error_translator)

    response = TestClient(app).post(
        "/v1/control/list-runs",
        content=ListRunsRequest().SerializeToString(),
        headers={"content-type": PROTOBUF_MEDIA_TYPE},
    )

    assert response.status_code == 500
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {
        "detail": "Invalid protobuf response.",
        "code": ApiErrorCode.INVALID_PROTOBUF_RESPONSE.value,
    }
    assert b"Conflict" not in response.content


def test_non_protobuf_request_in_state_returns_internal_error() -> None:
    """The protobuf request dependency rejects a non-protobuf state value."""
    app = FastAPI()

    @app.post("/v1/control/list-runs")
    def list_runs(request: Request) -> Response:
        request.state.protobuf_request = object()
        _ = get_protobuf_request(request)
        return Response()

    app.middleware("http")(http_error_translator)

    response = TestClient(app).post("/v1/control/list-runs")

    assert response.status_code == 500
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {
        "detail": "Invalid protobuf request.",
        "code": ApiErrorCode.INVALID_PROTOBUF_REQUEST.value,
    }


@pytest.mark.parametrize(
    ("proto_request", "expected_limit", "expected_skip"),
    [
        (ListRunsRequest(), 20, 0),
        (ListRunsRequest(limit=1), 1, 0),
        (ListRunsRequest(limit=1, skip=2), 1, 2),
    ],
)
def test_list_runs_returns_runs_from_linkstate(
    proto_request: ListRunsRequest, expected_limit: int, expected_skip: int
) -> None:
    """ListRuns applies the default and requested pagination."""
    linkstate = Mock(spec=LinkState)
    run = Run.create_empty(7)
    run.flwr_aid = _ACCOUNT.flwr_aid
    linkstate.get_run_info.return_value = [run]
    app = _create_app()
    app.dependency_overrides[get_linkstate] = lambda: linkstate
    client = TestClient(app)

    response = client.post(
        "/v1/control/list-runs",
        content=proto_request.SerializeToString(),
        headers={
            "authorization": "Bearer access-token",
            "content-type": PROTOBUF_MEDIA_TYPE,
        },
    )
    proto_response = ListRunsResponse.FromString(response.content)

    assert response.status_code == 200
    assert response.headers["content-type"] == PROTOBUF_MEDIA_TYPE
    assert set(proto_response.run_dict) == {7}
    assert proto_response.run_dict[7].account_name == _ACCOUNT.account_name
    assert datetime.fromisoformat(proto_response.now)
    assert linkstate.get_run_info.call_args_list == [
        call(
            flwr_aids=[_ACCOUNT.flwr_aid],
            order_by="pending_at",
            ascending=False,
            limit=expected_limit,
            skip=expected_skip,
        )
    ]


def test_list_runs_cleans_finished_runs_on_page() -> None:
    """ListRuns cleans up finished runs returned on the page."""
    linkstate = Mock(spec=LinkState)
    page_run = Run.create_empty(7)
    page_run.flwr_aid = _ACCOUNT.flwr_aid
    page_run.status.status = Status.FINISHED
    linkstate.get_run_info.return_value = [page_run]
    app = _create_app()
    app.dependency_overrides[get_linkstate] = lambda: linkstate

    response = TestClient(app).post(
        "/v1/control/list-runs",
        content=ListRunsRequest(limit=1).SerializeToString(),
        headers={
            "authorization": "Bearer access-token",
            "content-type": PROTOBUF_MEDIA_TYPE,
        },
    )

    assert response.status_code == 200
    assert set(ListRunsResponse.FromString(response.content).run_dict) == {7}
    linkstate.cleanup_run.assert_called_once_with(7)


def test_list_runs_rejects_invalid_token_without_refresh() -> None:
    """Control HTTP rejects invalid access tokens without refreshing them."""
    linkstate = Mock(spec=LinkState)
    authn_plugin = Mock()
    linkstate.get_run_info.return_value = []
    app = _create_app(authn_plugin=authn_plugin)
    authn_plugin.validate_tokens_in_metadata.return_value = (False, None)
    app.dependency_overrides[get_linkstate] = lambda: linkstate
    response = TestClient(app).post(
        "/v1/control/list-runs",
        content=ListRunsRequest().SerializeToString(),
        headers={
            "authorization": "Bearer invalid-token",
            "content-type": PROTOBUF_MEDIA_TYPE,
        },
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Not authenticated"}
    assert response.headers["www-authenticate"] == "Bearer"
    assert "x-access-token" not in response.headers
    assert "x-refresh-token" not in response.headers
    authn_plugin.refresh_tokens.assert_not_called()


def test_list_runs_rejects_non_protobuf_payload() -> None:
    """The protobuf translation middleware validates configured request bodies."""
    linkstate = Mock(spec=LinkState)
    app = _create_app()
    app.dependency_overrides[get_linkstate] = lambda: linkstate
    response = TestClient(app).post(
        "/v1/control/list-runs",
        content=b"{}",
        headers={
            "authorization": "Bearer access-token",
            "content-type": "application/json",
        },
    )

    assert response.status_code == 415
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {
        "detail": "Unsupported Content-Type.",
        "code": ApiErrorCode.UNSUPPORTED_CONTENT_TYPE.value,
    }


def test_list_runs_rejects_invalid_protobuf_bytes() -> None:
    """Return the shared JSON error for an invalid serialized protobuf."""
    linkstate = Mock(spec=LinkState)
    app = _create_app()
    app.dependency_overrides[get_linkstate] = lambda: linkstate

    response = TestClient(app).post(
        "/v1/control/list-runs",
        content=b"\x80",
        headers={
            "accept": PROTOBUF_MEDIA_TYPE,
            "authorization": "Bearer access-token",
            "content-type": PROTOBUF_MEDIA_TYPE,
        },
    )

    assert response.status_code == 400
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {
        "detail": "Invalid protobuf payload.",
        "code": ApiErrorCode.INVALID_PROTOBUF_PAYLOAD.value,
    }


def test_stream_logs_returns_framed_protobuf_responses() -> None:
    """Serialize every log response as one length-delimited stream frame."""
    linkstate = Mock(spec=LinkState)
    expected = [
        StreamLogsResponse(log_output="first\n", latest_timestamp=1.0),
        StreamLogsResponse(log_output="second\n", latest_timestamp=2.0),
    ]
    app = _create_app()
    app.dependency_overrides[get_linkstate] = lambda: linkstate

    def stream(
        request: StreamLogsRequest,
        account: AccountInfo,
        state: LinkState,
        is_active: Callable[[], bool],
    ) -> object:
        assert request == StreamLogsRequest(run_id=7, after_timestamp=0.5)
        assert account is _ACCOUNT
        assert state is linkstate
        assert is_active()
        return iter(expected)

    with patch(
        "flwr.superlink.routers.control.router.control_handlers.stream_logs",
        side_effect=stream,
    ):
        response = TestClient(app).post(
            "/v1/control/stream-logs",
            content=StreamLogsRequest(
                run_id=7, after_timestamp=0.5
            ).SerializeToString(),
            headers={
                "authorization": "Bearer access-token",
                "content-type": PROTOBUF_MEDIA_TYPE,
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == PROTOBUF_STREAM_MEDIA_TYPE
    assert response.content == b"".join(frame_message(message) for message in expected)


def test_stream_run_events_returns_framed_protobuf_responses() -> None:
    """Serialize every task event as one length-delimited stream frame."""
    linkstate = Mock(spec=LinkState)
    expected = [
        StreamRunEventsResponse(task_event=TaskEvent(id=5, run_id=7, event="first")),
        StreamRunEventsResponse(task_event=TaskEvent(id=6, run_id=7, event="second")),
    ]
    app = _create_app()
    app.dependency_overrides[get_linkstate] = lambda: linkstate

    with patch(
        "flwr.superlink.routers.control.router.control_handlers.stream_run_events",
        return_value=iter(expected),
    ) as stream:
        response = TestClient(app).post(
            "/v1/control/stream-run-events",
            content=StreamRunEventsRequest(
                run_id=7, after_task_event_id=4
            ).SerializeToString(),
            headers={
                "authorization": "Bearer access-token",
                "content-type": PROTOBUF_MEDIA_TYPE,
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == PROTOBUF_STREAM_MEDIA_TYPE
    assert response.content == b"".join(frame_message(message) for message in expected)
    request, account, state, is_active = stream.call_args.args
    assert request == StreamRunEventsRequest(run_id=7, after_task_event_id=4)
    assert account is _ACCOUNT
    assert state is linkstate
    assert callable(is_active)


def test_stream_logs_returns_missing_run_error_before_streaming() -> None:
    """Return a JSON error instead of opening a stream for an unknown run."""
    linkstate = Mock(spec=LinkState)
    linkstate.get_run_info.return_value = []
    app = _create_app()
    app.dependency_overrides[get_linkstate] = lambda: linkstate

    response = TestClient(app).post(
        "/v1/control/stream-logs",
        content=StreamLogsRequest(run_id=7).SerializeToString(),
        headers={
            "authorization": "Bearer access-token",
            "content-type": PROTOBUF_MEDIA_TYPE,
        },
    )

    assert response.status_code == 404
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {
        "detail": "Run ID not found.",
        "code": ApiErrorCode.RUN_ID_NOT_FOUND.value,
    }


def test_pull_artifacts_returns_provider_url() -> None:
    """Return the configured provider's URL for a finished owned run."""
    linkstate = Mock(spec=LinkState)
    run = Run.create_empty(7)
    run.flwr_aid = _ACCOUNT.flwr_aid
    run.status = RunStatus(Status.FINISHED, SubStatus.COMPLETED, "")
    linkstate.get_run_info.return_value = [run]
    artifact_provider = Mock(spec=ArtifactProvider)
    artifact_provider.get_url.return_value = "https://artifacts.example/run-7.zip"
    app = _create_app()
    app.state.artifact_provider = artifact_provider
    app.dependency_overrides[get_linkstate] = lambda: linkstate

    response = TestClient(app).post(
        "/v1/control/pull-artifacts",
        content=PullArtifactsRequest(run_id=7).SerializeToString(),
        headers={
            "authorization": "Bearer access-token",
            "content-type": PROTOBUF_MEDIA_TYPE,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == PROTOBUF_MEDIA_TYPE
    assert PullArtifactsResponse.FromString(response.content) == PullArtifactsResponse(
        url="https://artifacts.example/run-7.zip"
    )
    artifact_provider.get_url.assert_called_once_with(7)


def test_stream_run_events_returns_missing_run_error_before_streaming() -> None:
    """Return a JSON error instead of opening an event stream for an unknown run."""
    linkstate = Mock(spec=LinkState)
    linkstate.get_run_info.return_value = []
    app = _create_app()
    app.dependency_overrides[get_linkstate] = lambda: linkstate

    response = TestClient(app).post(
        "/v1/control/stream-run-events",
        content=StreamRunEventsRequest(run_id=7).SerializeToString(),
        headers={
            "authorization": "Bearer access-token",
            "content-type": PROTOBUF_MEDIA_TYPE,
        },
    )

    assert response.status_code == 404
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {
        "detail": "Run ID not found.",
        "code": ApiErrorCode.RUN_ID_NOT_FOUND.value,
    }


def test_pull_artifacts_returns_error_without_provider() -> None:
    """Return the established error when no artifact provider is configured."""
    app = _create_app()
    app.dependency_overrides[get_linkstate] = lambda: Mock(spec=LinkState)

    response = TestClient(app).post(
        "/v1/control/pull-artifacts",
        content=PullArtifactsRequest(run_id=7).SerializeToString(),
        headers={
            "authorization": "Bearer access-token",
            "content-type": PROTOBUF_MEDIA_TYPE,
        },
    )

    assert response.status_code == 501
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {
        "detail": "ControlServicer initialized without artifact provider.",
        "code": ApiErrorCode.NO_ARTIFACT_PROVIDER.value,
    }
