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

from fastapi import FastAPI
from fastapi.testclient import TestClient

from flwr.common.constant import NOOP_FLWR_AID
from flwr.common.event_log_plugin import EventLogWriterPlugin
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListRunsRequest,
    ListRunsResponse,
)
from flwr.server.superlink.linkstate import LinkState
from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.error import http_error_translator
from flwr.supercore.license_plugin import LicensePlugin
from flwr.supercore.protobuf.constants import PROTOBUF_MEDIA_TYPE
from flwr.supercore.run import Run
from flwr.superlink.dependencies.account import AccountAccessDependency
from flwr.superlink.dependencies.linkstate import get_linkstate
from flwr.superlink.routers.control.router import router


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


def test_list_runs_checks_license_and_logs_event() -> None:
    """ListRuns applies the Control license and event log integrations."""
    linkstate = Mock(spec=LinkState)
    authn_plugin = Mock()
    authz_plugin = Mock()
    license_plugin = Mock(spec=LicensePlugin)
    event_log_plugin = Mock(spec=EventLogWriterPlugin)
    account = AccountInfo(flwr_aid=NOOP_FLWR_AID, account_name="account")
    authn_plugin.validate_tokens_in_metadata.return_value = (True, account)
    authz_plugin.authorize.return_value = True
    license_plugin.check_license.return_value = True
    linkstate.get_run_info.return_value = []
    app = FastAPI()
    app.state.account_access_dep = AccountAccessDependency(authn_plugin, authz_plugin)
    app.state.control_license_plugin = license_plugin
    app.state.control_event_log_plugin = event_log_plugin
    app.include_router(router)
    app.dependency_overrides[get_linkstate] = lambda: linkstate

    response = TestClient(app).post(
        "/control/list-runs",
        content=ListRunsRequest().SerializeToString(),
        headers={"content-type": PROTOBUF_MEDIA_TYPE},
    )

    assert response.status_code == 200
    license_plugin.check_license.assert_called_once_with()
    before_args = event_log_plugin.compose_log_before_event.call_args.kwargs
    after_args = event_log_plugin.compose_log_after_event.call_args.kwargs
    assert before_args["request"] == ListRunsRequest()
    assert before_args["account_info"] == account
    assert before_args["method_name"] == "/flwr.proto.Control/ListRuns"
    assert after_args["account_info"] == account
    assert after_args["method_name"] == "/flwr.proto.Control/ListRuns"
    assert event_log_plugin.write_log.call_count == 2


def test_list_runs_rejects_an_invalid_license() -> None:
    """ListRuns does not call its handler when the license check fails."""
    license_plugin = Mock(spec=LicensePlugin)
    event_log_plugin = Mock(spec=EventLogWriterPlugin)
    license_plugin.check_license.return_value = False
    authn_plugin = Mock()
    authz_plugin = Mock()
    account = AccountInfo(flwr_aid=NOOP_FLWR_AID, account_name="account")
    authn_plugin.validate_tokens_in_metadata.return_value = (True, account)
    authz_plugin.authorize.return_value = True
    app = FastAPI()
    app.state.account_access_dep = AccountAccessDependency(authn_plugin, authz_plugin)
    app.state.control_license_plugin = license_plugin
    app.state.control_event_log_plugin = event_log_plugin
    app.include_router(router)
    app.dependency_overrides[get_linkstate] = lambda: Mock(spec=LinkState)
    app.middleware("http")(http_error_translator)

    response = TestClient(app).post(
        "/control/list-runs",
        content=ListRunsRequest().SerializeToString(),
        headers={"content-type": PROTOBUF_MEDIA_TYPE},
    )

    assert response.status_code == 403
    license_plugin.check_license.assert_called_once_with()
    event_log_plugin.write_log.assert_not_called()
