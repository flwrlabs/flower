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


from collections import Counter
from collections.abc import Callable
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from google.protobuf.message import Message

from flwr.common.constant import (
    ACCESS_TOKEN_KEY,
    NOOP_FLWR_AID,
    REFRESH_TOKEN_KEY,
    Status,
    SubStatus,
)
from flwr.proto import control_pb2
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    AddAppRequest,
    AddAppResponse,
    BeginConnectorOAuthRequest,
    BeginConnectorOAuthResponse,
    CompleteConnectorOAuthRequest,
    CompleteConnectorOAuthResponse,
    Connector,
    DisconnectConnectorRequest,
    DisconnectConnectorResponse,
    GetAuthTokensRequest,
    GetAuthTokensResponse,
    GetLoginDetailsRequest,
    GetLoginDetailsResponse,
    ListConnectorsRequest,
    ListConnectorsResponse,
    ListRunsRequest,
    ListRunsResponse,
    PullArtifactsRequest,
    PullArtifactsResponse,
    RefreshAuthTokensRequest,
    RefreshAuthTokensResponse,
    StartRunRequest,
    StartRunResponse,
    StreamLogsRequest,
    StreamLogsResponse,
    StreamRunEventsRequest,
    StreamRunEventsResponse,
)
from flwr.proto.task_pb2 import TaskEvent  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkState
from flwr.supercore.auth.typing import (
    AccountAuthCredentials,
    AccountAuthLoginDetails,
    AccountInfo,
)
from flwr.supercore.error import ApiErrorCode, http_error_translator
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
from flwr.superlink.routers.control.middlewares import ControlAuthenticationMiddleware
from flwr.superlink.routers.control.router import add_app as add_app_route
from flwr.superlink.routers.control.router import router
from flwr.superlink.routers.control.router import start_run as start_run_route
from flwr.superlink.servicer.control import control_handlers

_ACCOUNT = AccountInfo(flwr_aid=NOOP_FLWR_AID, account_name="account")


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


def test_control_http_routes_cover_all_grpc_methods() -> None:
    """Expose every Control gRPC request type plus HTTP-only token refresh."""
    grpc_request_types = Counter(
        method.input_type.full_name
        for method in control_pb2.DESCRIPTOR.services_by_name["Control"].methods
    )
    http_request_types = Counter(
        request_type.DESCRIPTOR.full_name
        for (method, path), request_type in PROTOBUF_REQUEST_TYPES.items()
        if method == "POST" and path.startswith("/v1/control/")
    )
    grpc_request_types[RefreshAuthTokensRequest.DESCRIPTOR.full_name] += 1

    assert http_request_types == grpc_request_types


@pytest.mark.parametrize(
    ("path", "protobuf_request", "parse_response", "expected", "handler_name"),
    [
        (
            "/v1/control/list-connectors",
            ListConnectorsRequest(federation="agent"),
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
            DisconnectConnectorRequest(connector_ref="google-drive"),
            DisconnectConnectorResponse.FromString,
            DisconnectConnectorResponse(),
            "disconnect_connector",
        ),
        (
            "/v1/control/begin-connector-oauth",
            BeginConnectorOAuthRequest(
                connector_ref="google-drive",
                redirect_uri="https://example.test/oauth/callback",
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
        content=BeginConnectorOAuthRequest().SerializeToString(),
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
def test_connector_routes_reject_invalid_access_tokens_without_refresh(
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
        response = start_run_route(request, linkstate, _ACCOUNT, "grpc-rere", "unknown")

    assert response is expected
    start_run.assert_called_once_with(
        request,
        _ACCOUNT,
        linkstate,
        "grpc-rere",
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
            fleet_api_type="grpc-rere",
            run_source="cli",
        )

    assert start_run.call_args.kwargs["source"] == "cli"


def test_add_app_forwards_fleet_api_type() -> None:
    """Forward the configured Fleet API transport type to the control handler."""
    request = AddAppRequest()
    linkstate = Mock()
    expected = AddAppResponse()

    with patch(
        "flwr.superlink.routers.control.router.control_handlers.add_app",
        return_value=expected,
    ) as add_app:
        response = add_app_route(request, linkstate, _ACCOUNT, "grpc-rere")

    assert response is expected
    add_app.assert_called_once_with(request, _ACCOUNT, linkstate, "grpc-rere")


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


def test_list_runs_returns_runs_from_linkstate() -> None:
    """ListRuns serializes the runs returned by LinkState."""
    linkstate = Mock(spec=LinkState)
    run = Run.create_empty(7)
    run.flwr_aid = _ACCOUNT.flwr_aid
    linkstate.get_run_info.return_value = [run]
    app = _create_app()
    app.dependency_overrides[get_linkstate] = lambda: linkstate
    client = TestClient(app)

    response = client.post(
        "/v1/control/list-runs",
        content=ListRunsRequest(limit=1).SerializeToString(),
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
    linkstate.get_run_info.assert_called_once_with(
        flwr_aids=[_ACCOUNT.flwr_aid],
        order_by="pending_at",
        ascending=False,
        limit=1,
    )


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
