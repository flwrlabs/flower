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
"""Fleet protobuf HTTP route tests."""

from base64 import b64encode
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from flwr.common.constant import TIMESTAMP_HEADER
from flwr.common.event_log_plugin import EventLogWriterPlugin
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    ActivateNodeRequest,
    ActivateNodeResponse,
    RegisterNodeFleetRequest,
    RegisterNodeFleetResponse,
)
from flwr.proto.heartbeat_pb2 import (  # pylint: disable=E0611
    SendNodeHeartbeatRequest,
    SendNodeHeartbeatResponse,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.supercore.constant import (
    FLEET_HTTP_PUBLIC_KEY_HEADER,
    FLEET_HTTP_SIGNATURE_HEADER,
    FLWR_PACKAGE_NAME_METADATA_KEY,
)
from flwr.supercore.date import now
from flwr.supercore.error import ApiErrorCode
from flwr.supercore.primitives.asymmetric import (
    generate_key_pairs,
    public_key_to_bytes,
    sign_message,
)
from flwr.supercore.protobuf.constants import PROTOBUF_MEDIA_TYPE
from flwr.superlink import extensions, main
from flwr.superlink.routers.control import middlewares as control_middlewares
from flwr.superlink.servicer.fleet import fleet_handlers

from . import node_auth
from .router import router as fleet_router


def test_fleet_http_handlers_and_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fleet HTTP uses shared handlers and rejects unsigned node calls."""
    monkeypatch.setattr(extensions, "get_middleware", lambda: ())
    monkeypatch.setattr(extensions, "configure_app", lambda _: None)
    monkeypatch.setattr(control_middlewares, "get_license_plugin", lambda: None)
    app = main.create_app()
    app.include_router(fleet_router)
    control_plugin = Mock(spec=EventLogWriterPlugin)
    app.state.control_event_log_plugin = control_plugin
    client = TestClient(app)

    private_key, public_key = generate_key_pairs()
    public_key_bytes = public_key_to_bytes(public_key)
    timestamp = now().isoformat()
    headers = {
        "content-type": PROTOBUF_MEDIA_TYPE,
        FLEET_HTTP_PUBLIC_KEY_HEADER: b64encode(public_key_bytes).decode("ascii"),
        FLEET_HTTP_SIGNATURE_HEADER: b64encode(
            sign_message(private_key, timestamp.encode("ascii"))
        ).decode("ascii"),
        TIMESTAMP_HEADER: timestamp,
        FLWR_PACKAGE_NAME_METADATA_KEY: "flwr",
    }

    response = client.post(
        "/v1/fleet/register-node",
        content=RegisterNodeFleetRequest(
            public_key=public_key_bytes
        ).SerializeToString(),
        headers=headers,
    )
    assert response.status_code == 200
    node_id = RegisterNodeFleetResponse.FromString(response.content).node_id
    assert node_id > 0

    response = client.post(
        "/v1/fleet/activate-node",
        content=ActivateNodeRequest(
            public_key=public_key_bytes, heartbeat_interval=30
        ).SerializeToString(),
        headers=headers,
    )
    assert response.status_code == 200
    assert ActivateNodeResponse.FromString(response.content).node_id == node_id

    heartbeat = SendNodeHeartbeatRequest(
        node=Node(node_id=node_id), heartbeat_interval=30
    )
    response = client.post(
        "/v1/fleet/send-node-heartbeat",
        content=heartbeat.SerializeToString(),
        headers=headers,
    )
    assert response.status_code == 200
    assert SendNodeHeartbeatResponse.FromString(response.content).success
    control_plugin.write_log.assert_not_called()

    response = client.post(
        "/v1/fleet/send-node-heartbeat",
        content=SendNodeHeartbeatRequest(
            node=Node(node_id=node_id), heartbeat_interval=1
        ).SerializeToString(),
        headers=headers,
    )
    assert response.status_code == 400
    assert response.json()["code"] == ApiErrorCode.FLEET_INVALID_HEARTBEAT_INTERVAL

    response = client.post(
        "/v1/fleet/send-node-heartbeat",
        content=heartbeat.SerializeToString(),
        headers={**headers, FLEET_HTTP_SIGNATURE_HEADER: "bad"},
    )
    assert response.status_code == 401

    response = client.post(
        "/v1/fleet/send-node-heartbeat",
        content=SendNodeHeartbeatRequest(
            node=Node(node_id=node_id + 1), heartbeat_interval=30
        ).SerializeToString(),
        headers=headers,
    )
    assert response.status_code == 401


def test_fleet_http_event_log(monkeypatch: pytest.MonkeyPatch) -> None:
    """Log Fleet calls before and after the handler when a writer is provided."""
    monkeypatch.setattr(extensions, "get_middleware", lambda: ())
    monkeypatch.setattr(extensions, "configure_app", lambda _: None)
    fleet_plugin = Mock(spec=EventLogWriterPlugin)
    expected = RegisterNodeFleetResponse(node_id=42)
    monkeypatch.setattr(fleet_handlers, "register_node", lambda **_: expected)

    app = main.create_app()
    app.state.fleet_event_log_plugin = fleet_plugin
    app.include_router(fleet_router)
    monkeypatch.setattr(node_auth, "authenticate_node", lambda _: None)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/v1/fleet/register-node",
        content=RegisterNodeFleetRequest().SerializeToString(),
        headers={"content-type": PROTOBUF_MEDIA_TYPE},
    )

    assert response.status_code == 200
    assert fleet_plugin.write_log.call_count == 2
    assert fleet_plugin.compose_log_before_event.call_args.kwargs["method_name"] == (
        "/v1/fleet/register-node"
    )
    assert fleet_plugin.compose_log_after_event.call_args.kwargs["response"] == expected

    client.get("/health")
    assert fleet_plugin.write_log.call_count == 2

    def fail(**_: object) -> RegisterNodeFleetResponse:
        raise RuntimeError("handler failed")

    monkeypatch.setattr(fleet_handlers, "register_node", fail)
    response = client.post(
        "/v1/fleet/register-node",
        content=RegisterNodeFleetRequest().SerializeToString(),
        headers={"content-type": PROTOBUF_MEDIA_TYPE},
    )
    assert response.status_code == 500
    assert isinstance(
        fleet_plugin.compose_log_after_event.call_args.kwargs["response"],
        RuntimeError,
    )
    assert fleet_plugin.write_log.call_count == 4
