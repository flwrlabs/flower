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


from collections.abc import Callable
from datetime import datetime
from unittest.mock import Mock, patch

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from flwr.common.constant import NOOP_FLWR_AID, Status, SubStatus
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListRunsRequest,
    ListRunsResponse,
    PullArtifactsRequest,
    PullArtifactsResponse,
    StartRunRequest,
    StartRunResponse,
    StreamLogsRequest,
    StreamLogsResponse,
    StreamRunEventsRequest,
    StreamRunEventsResponse,
)
from flwr.proto.task_pb2 import TaskEvent  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkState
from flwr.supercore.auth.typing import AccountInfo
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
from flwr.superlink.routers.control.router import router
from flwr.superlink.routers.control.router import start_run as start_run_route

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


def test_start_run_forwards_resolved_source() -> None:
    """Forward the normalized source to the control handler."""
    request = StartRunRequest()
    linkstate = Mock()
    expected = StartRunResponse(run_id=1)

    with patch(
        "flwr.superlink.routers.control.router.control_handlers.start_run",
        return_value=expected,
    ) as start_run:
        response = start_run_route(request, linkstate, _ACCOUNT, "unknown")

    assert response is expected
    start_run.assert_called_once_with(
        request,
        _ACCOUNT,
        linkstate,
        "",
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
        start_run_route(request, linkstate, _ACCOUNT, run_source="cli")

    assert start_run.call_args.kwargs["source"] == "cli"


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
