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

from fastapi import FastAPI
from fastapi.testclient import TestClient

from flwr.proto.control_pb2 import ListRunsRequest  # pylint: disable=E0611
from flwr.supercore.protobuf.constants import PROTOBUF_MEDIA_TYPE
from flwr.superlink.routers.control.router import router


def test_list_runs_is_a_protobuf_endpoint() -> None:
    """The ListRuns placeholder accepts a protobuf request."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.post(
        "/control/rpc/ListRuns",
        content=ListRunsRequest().SerializeToString(),
        headers={"content-type": PROTOBUF_MEDIA_TYPE},
    )

    assert response.status_code == 501
