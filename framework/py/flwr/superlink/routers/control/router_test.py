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


from datetime import datetime
from unittest.mock import Mock

from fastapi import FastAPI, Request, Response
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from flwr.common.constant import NOOP_FLWR_AID
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListRunsRequest,
    ListRunsResponse,
)
from flwr.server.superlink.linkstate import LinkState
from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.error import ApiErrorCode, http_error_translator
from flwr.supercore.protobuf.constants import PROTOBUF_MEDIA_TYPE
from flwr.supercore.protobuf.translation import (
    PROTOBUF_REQUEST_TYPES,
    ProtobufTranslationMiddleware,
    get_protobuf_request,
)
from flwr.supercore.run import Run
from flwr.superlink.dependencies.account import AccountAccessDependency
from flwr.superlink.dependencies.linkstate import get_linkstate
from flwr.superlink.routers.control.middlewares import ControlAuthenticationMiddleware
from flwr.superlink.routers.control.router import router

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

    assert route_keys == set(PROTOBUF_REQUEST_TYPES)


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
    assert response.json()["code"] == ApiErrorCode.INVALID_PROTOBUF_RESPONSE


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
    assert response.json()["code"] == ApiErrorCode.INVALID_PROTOBUF_REQUEST


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
    assert response.json() == {
        "code": ApiErrorCode.ACCOUNT_AUTHENTICATION_FAILED,
        "public_message": "Authentication failed.",
        "public_details": None,
    }
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
    assert response.json()["code"] == ApiErrorCode.UNSUPPORTED_CONTENT_TYPE
