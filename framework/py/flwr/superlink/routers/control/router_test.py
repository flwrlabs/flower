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

from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListRunsRequest,
    ListRunsResponse,
)
from flwr.server.superlink.linkstate import LinkState
from flwr.supercore.protobuf.constants import PROTOBUF_MEDIA_TYPE
from flwr.supercore.run import Run
from flwr.superlink.dependencies.linkstate import get_linkstate
from flwr.superlink.routers.control.router import router


def test_list_runs_returns_runs_from_linkstate() -> None:
    """ListRuns serializes the runs returned by LinkState."""
    linkstate = Mock(spec=LinkState)
    linkstate.get_run_info.return_value = [Run.create_empty(7)]
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_linkstate] = lambda: linkstate
    client = TestClient(app)

    response = client.post(
        "/control/rpc/ListRuns",
        content=ListRunsRequest(limit=1).SerializeToString(),
        headers={"content-type": PROTOBUF_MEDIA_TYPE},
    )
    proto_response = ListRunsResponse.FromString(response.content)

    assert response.status_code == 200
    assert set(proto_response.run_dict) == {7}
    assert datetime.fromisoformat(proto_response.now)
    linkstate.get_run_info.assert_called_once_with(
        order_by="pending_at",
        ascending=False,
        limit=1,
    )
