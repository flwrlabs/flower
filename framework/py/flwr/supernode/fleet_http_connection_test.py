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
"""Tests for the SuperNode Fleet HTTP connection."""

import os
import signal
from collections.abc import Callable
from contextlib import nullcontext
from unittest.mock import Mock

import httpx
from pytest import MonkeyPatch, raises

from flwr.app.message import Message

# pylint: disable=E0611
from flwr.proto.fab_pb2 import GetFabResponse  # pylint: disable=E0611
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    ActivateNodeResponse,
    PullMessagesResponse,
    PushMessagesResponse,
)
from flwr.proto.message_pb2 import ConfirmMessageReceivedResponse
from flwr.proto.message_pb2 import Message as ProtoMessage  # pylint: disable=E0611
from flwr.proto.message_pb2 import ObjectTree, PullObjectResponse, PushObjectResponse
from flwr.proto.run_pb2 import GetRunResponse  # pylint: disable=E0611

# pylint: enable=E0611
from flwr.supercore.interceptors.http.runtime_version import (
    RuntimeVersionHttpInterceptor,
)
from flwr.supercore.primitives.asymmetric import generate_key_pairs

from . import fleet_http_connection as connection
from .fleet_http_client import FleetHttpClient
from .node_auth_http_interceptor import NodeAuthHttpInterceptor


def test_http_connection_provides_worker_callbacks(  # pylint: disable=R0914,R0915
    monkeypatch: MonkeyPatch,
) -> None:
    """Map worker operations to Fleet HTTP calls and clean up the node."""
    client = Mock(spec=FleetHttpClient)
    client.ActivateNode.return_value = ActivateNodeResponse(node_id=42)
    client.PullMessages.side_effect = [
        PullMessagesResponse(),
        PullMessagesResponse(
            messages_list=[ProtoMessage()], message_object_trees=[ObjectTree()]
        ),
    ]
    client.PushMessages.return_value = PushMessagesResponse(
        objects_to_push=["object"], session_id="session"
    )
    client.GetRun.return_value = GetRunResponse()
    client.GetFab.return_value = GetFabResponse()
    client.PullObject.return_value = PullObjectResponse(
        object_found=True, object_available=True, object_content=b"content"
    )
    client.PushObject.return_value = PushObjectResponse(stored=True)
    client.ConfirmMessageReceived.return_value = ConfirmMessageReceivedResponse()
    client.SendNodeHeartbeat.return_value.success = True
    factory = Mock(return_value=nullcontext(client))
    monkeypatch.setattr(FleetHttpClient, "from_server_address", factory)
    heartbeat_sender = Mock(is_running=True)
    heartbeat_fns: list[Callable[[], bool]] = []

    def make_heartbeat_sender(fn: Callable[[], bool]) -> Mock:
        heartbeat_fns.append(fn)
        return heartbeat_sender

    monkeypatch.setattr(connection, "HeartbeatSender", make_heartbeat_sender)
    message = Mock(spec=Message)
    message.has_content.return_value = False
    monkeypatch.setattr(connection, "message_to_proto", lambda _: ProtoMessage())
    monkeypatch.setattr(connection, "message_from_proto", lambda _: message)
    run = Mock()
    fab = Mock()
    monkeypatch.setattr(connection, "run_from_proto", lambda _: run)
    monkeypatch.setattr(connection, "fab_from_proto", lambda _: fab)

    with connection.http_request_response(
        "fleet.example:8080", insecure=True, max_retries=0
    ) as callbacks:
        node_id, receive, send, get_run, get_fab, pull, push, confirm = callbacks
        assert node_id == 42
        assert receive() is None
        assert receive() == (message, ObjectTree())
        assert send(message, ObjectTree(), 1.0) == ({"object"}, "session")
        assert get_run(3) is run
        assert get_fab("hash", 3) is fab
        assert pull(3, "object") == b"content"
        push(3, "session", "object", b"content")
        confirm(3, "object")
        assert heartbeat_fns[0]() is True
        assert client.SendNodeHeartbeat.call_args.args[0].node.node_id == 42
        client.SendNodeHeartbeat.side_effect = httpx.ConnectError("unavailable")
        assert heartbeat_fns[0]() is False
        client.SendNodeHeartbeat.side_effect = httpx.HTTPStatusError(
            "unauthorized",
            request=httpx.Request("POST", "https://fleet.example"),
            response=httpx.Response(401),
        )
        kill = Mock(side_effect=SystemExit)
        monkeypatch.setattr(os, "kill", kill)
        with raises(SystemExit):
            heartbeat_fns[0]()
        kill.assert_called_once_with(os.getpid(), signal.SIGINT)

    assert client.RegisterNode.call_count == 1
    assert client.ActivateNode.call_count == 1
    assert client.DeactivateNode.call_count == 1
    assert client.UnregisterNode.call_count == 1
    heartbeat_sender.start.assert_called_once_with()
    heartbeat_sender.stop.assert_called_once_with()
    assert isinstance(
        factory.call_args.kwargs["interceptors"][0], RuntimeVersionHttpInterceptor
    )
    assert isinstance(
        factory.call_args.kwargs["interceptors"][1], NodeAuthHttpInterceptor
    )


def test_http_connection_preserves_managed_identity(monkeypatch: MonkeyPatch) -> None:
    """Skip registration and unregistration for configured identity keys."""
    client = Mock(spec=FleetHttpClient)
    client.ActivateNode.return_value = ActivateNodeResponse(node_id=42)
    monkeypatch.setattr(
        FleetHttpClient, "from_server_address", Mock(return_value=nullcontext(client))
    )
    monkeypatch.setattr(connection, "HeartbeatSender", lambda _: Mock(is_running=True))

    with connection.http_request_response(
        "fleet.example:8080", insecure=False, authentication_keys=generate_key_pairs()
    ):
        pass

    client.RegisterNode.assert_not_called()
    client.UnregisterNode.assert_not_called()


def test_http_connection_unregisters_after_failed_deactivation(
    monkeypatch: MonkeyPatch,
) -> None:
    """Unregister a self-registered node even if it is already offline."""
    client = Mock(spec=FleetHttpClient)
    client.ActivateNode.return_value = ActivateNodeResponse(node_id=42)
    client.DeactivateNode.side_effect = httpx.HTTPStatusError(
        "already offline",
        request=httpx.Request("POST", "https://fleet.example/v1/fleet/deactivate-node"),
        response=httpx.Response(400),
    )
    monkeypatch.setattr(
        FleetHttpClient, "from_server_address", Mock(return_value=nullcontext(client))
    )
    monkeypatch.setattr(connection, "HeartbeatSender", lambda _: Mock(is_running=False))

    with connection.http_request_response("fleet.example:8080", insecure=True):
        pass

    client.UnregisterNode.assert_called_once()
