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
from typing import cast
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest import MonkeyPatch
from starlette.middleware.base import BaseHTTPMiddleware

from flwr.common.constant import NOOP_FLWR_AID
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    GetLoginDetailsRequest,
    GetLoginDetailsResponse,
    ListRunsRequest,
    ListRunsResponse,
)
from flwr.server.superlink.linkstate import LinkState
from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.error import ApiErrorCode, http_error_translator
from flwr.supercore.protobuf.constants import PROTOBUF_MEDIA_TYPE
from flwr.supercore.protobuf.routing import ProtobufTranslationMiddleware
from flwr.supercore.run import Run
from flwr.superlink.dependencies.account import AccountAccessDependency
from flwr.superlink.dependencies.linkstate import get_linkstate
from flwr.superlink.routers.control.router import configure_middlewares, router
from flwr.superlink.servicer.control import control_handlers


def test_configure_middlewares_registers_required_outer_layers() -> None:
    """Error and protobuf translation are the two outermost middleware layers."""
    app = FastAPI()

    configure_middlewares(app)

    error_middleware, protobuf_middleware = app.user_middleware[:2]
    assert cast(object, error_middleware.cls) is BaseHTTPMiddleware
    assert error_middleware.kwargs["dispatch"] is http_error_translator
    assert cast(object, protobuf_middleware.cls) is ProtobufTranslationMiddleware


def test_list_runs_returns_runs_from_linkstate() -> None:
    """ListRuns serializes the runs returned by LinkState."""
    linkstate = Mock(spec=LinkState)
    authn_plugin = Mock()
    authz_plugin = Mock()
    account = AccountInfo(flwr_aid=NOOP_FLWR_AID, account_name="account")
    run = Run.create_empty(7)
    run.flwr_aid = account.flwr_aid
    linkstate.get_run_info.return_value = [run]
    authn_plugin.validate_tokens_in_metadata.return_value = (True, account)
    authz_plugin.authorize.return_value = True
    app = FastAPI()
    app.state.account_access_dep = AccountAccessDependency(authn_plugin, authz_plugin)
    app.include_router(router)
    configure_middlewares(app)
    app.dependency_overrides[get_linkstate] = lambda: linkstate
    client = TestClient(app)

    response = client.post(
        "/control/list-runs",
        content=ListRunsRequest(limit=1).SerializeToString(),
        headers={"content-type": PROTOBUF_MEDIA_TYPE},
    )
    proto_response = ListRunsResponse.FromString(response.content)

    assert response.status_code == 200
    assert set(proto_response.run_dict) == {7}
    assert proto_response.run_dict[7].account_name == account.account_name
    assert datetime.fromisoformat(proto_response.now)
    linkstate.get_run_info.assert_called_once_with(
        flwr_aids=[account.flwr_aid],
        order_by="pending_at",
        ascending=False,
        limit=1,
    )
    authz_plugin.authorize.assert_called_once_with(account)
    authn_plugin.validate_tokens_in_metadata.assert_called_once()


def test_list_runs_preserves_refreshed_authentication_tokens() -> None:
    """The authentication middleware adds refreshed tokens to protobuf responses."""
    linkstate = Mock(spec=LinkState)
    authn_plugin = Mock()
    authz_plugin = Mock()
    account = AccountInfo(flwr_aid=NOOP_FLWR_AID, account_name="account")
    authn_plugin.validate_tokens_in_metadata.return_value = (False, None)
    authn_plugin.refresh_tokens.return_value = (
        [("x-access-token", "new-access-token")],
        account,
    )
    authz_plugin.authorize.return_value = True
    linkstate.get_run_info.return_value = []
    app = FastAPI()
    app.state.account_access_dep = AccountAccessDependency(authn_plugin, authz_plugin)
    app.include_router(router)
    configure_middlewares(app)
    app.dependency_overrides[get_linkstate] = lambda: linkstate
    response = TestClient(app).post(
        "/control/list-runs",
        content=ListRunsRequest().SerializeToString(),
        headers={"content-type": PROTOBUF_MEDIA_TYPE},
    )

    assert response.status_code == 200
    assert response.headers["x-access-token"] == "new-access-token"
    assert response.headers.get_list("content-length") == [str(len(response.content))]


def test_list_runs_rejects_non_protobuf_payload() -> None:
    """The protobuf translation middleware validates configured request bodies."""
    linkstate = Mock(spec=LinkState)
    authn_plugin = Mock()
    authz_plugin = Mock()
    account = AccountInfo(flwr_aid=NOOP_FLWR_AID, account_name="account")
    authn_plugin.validate_tokens_in_metadata.return_value = (True, account)
    authz_plugin.authorize.return_value = True
    app = FastAPI()
    app.state.account_access_dep = AccountAccessDependency(authn_plugin, authz_plugin)
    app.include_router(router)
    configure_middlewares(app)
    app.dependency_overrides[get_linkstate] = lambda: linkstate
    response = TestClient(app).post(
        "/control/list-runs",
        content=b"{}",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 415
    assert response.json()["code"] == ApiErrorCode.UNSUPPORTED_CONTENT_TYPE


def test_get_login_details_does_not_require_authentication(
    monkeypatch: MonkeyPatch,
) -> None:
    """The login bootstrap endpoint remains available before authentication."""
    authn_plugin = Mock()
    authz_plugin = Mock()
    expected = GetLoginDetailsResponse(authn_type="noop")
    monkeypatch.setattr(
        control_handlers,
        "get_login_details",
        lambda _request, _plugin: expected,
    )
    app = FastAPI()
    app.state.account_access_dep = AccountAccessDependency(authn_plugin, authz_plugin)
    app.include_router(router)
    configure_middlewares(app)
    response = TestClient(app).post(
        "/control/get-login-details",
        content=GetLoginDetailsRequest().SerializeToString(),
        headers={"content-type": PROTOBUF_MEDIA_TYPE},
    )

    assert response.status_code == 200
    assert GetLoginDetailsResponse.FromString(response.content) == expected
    authn_plugin.validate_tokens_in_metadata.assert_not_called()
