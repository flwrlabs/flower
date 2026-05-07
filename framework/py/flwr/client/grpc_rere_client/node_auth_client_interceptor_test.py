# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Flower client interceptor tests."""


import threading
import unittest
from collections.abc import Callable, Sequence
from concurrent import futures
from typing import Any
from unittest.mock import Mock

import grpc
from google.protobuf.message import Message as GrpcMessage
from parameterized import parameterized

from flwr.client.grpc_rere_client.connection import grpc_request_response
from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.constant import PUBLIC_KEY_HEADER, SIGNATURE_HEADER, TIMESTAMP_HEADER
from flwr.common.message import Message
from flwr.common.record import RecordDict
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    ActivateNodeRequest,
    ActivateNodeResponse,
    DeactivateNodeRequest,
    DeactivateNodeResponse,
    PullMessagesRequest,
    PullMessagesResponse,
    PushMessagesRequest,
    PushMessagesResponse,
    RegisterNodeFleetRequest,
    RegisterNodeFleetResponse,
    UnregisterNodeFleetRequest,
    UnregisterNodeFleetResponse,
)
from flwr.proto.heartbeat_pb2 import (  # pylint: disable=E0611
    SendNodeHeartbeatRequest,
    SendNodeHeartbeatResponse,
)
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
from flwr.supercore.constant import (
    FLWR_COMPONENT_NAME_METADATA_KEY,
    FLWR_PACKAGE_NAME_METADATA_KEY,
    FLWR_PACKAGE_VERSION_METADATA_KEY,
)
from flwr.supercore.primitives.asymmetric import (
    generate_key_pairs,
    public_key_to_bytes,
    verify_signature,
)


class _MockServicer:
    """Mock Servicer for Flower clients."""

    def __init__(self) -> None:
        """Initialize mock servicer."""
        self._lock = threading.Lock()
        self._received_client_metadata: Sequence[tuple[str, str | bytes]] | None = None
        self._received_message_bytes: bytes = b""

    def unary_unary(  # pylint: disable=too-many-return-statements
        self, request: GrpcMessage, context: grpc.ServicerContext
    ) -> GrpcMessage:
        """Handle unary call."""
        with self._lock:
            self._received_client_metadata = context.invocation_metadata()
            self._received_message_bytes = request.SerializeToString(deterministic=True)

            if isinstance(request, RegisterNodeFleetRequest):
                return RegisterNodeFleetResponse()
            if isinstance(request, ActivateNodeRequest):
                return ActivateNodeResponse(node_id=123)
            if isinstance(request, DeactivateNodeRequest):
                return DeactivateNodeResponse()
            if isinstance(request, UnregisterNodeFleetRequest):
                return UnregisterNodeFleetResponse()
            if isinstance(request, PushMessagesRequest):
                return PushMessagesResponse()
            if isinstance(request, GetRunRequest):
                return GetRunResponse()
            if isinstance(request, SendNodeHeartbeatRequest):
                return SendNodeHeartbeatResponse(success=True)
            return PullMessagesResponse(messages_list=[])

    def received_client_metadata(
        self,
    ) -> Sequence[tuple[str, str | bytes]] | None:
        """Return received client metadata."""
        with self._lock:
            return self._received_client_metadata

    def received_message_bytes(self) -> bytes:
        """Return received message bytes."""
        with self._lock:
            return self._received_message_bytes


def _add_generic_handler(servicer: _MockServicer, server: grpc.Server) -> None:
    rpc_method_handlers = {
        "RegisterNode": grpc.unary_unary_rpc_method_handler(
            servicer.unary_unary,
            request_deserializer=RegisterNodeFleetRequest.FromString,
            response_serializer=RegisterNodeFleetResponse.SerializeToString,
        ),
        "ActivateNode": grpc.unary_unary_rpc_method_handler(
            servicer.unary_unary,
            request_deserializer=ActivateNodeRequest.FromString,
            response_serializer=ActivateNodeResponse.SerializeToString,
        ),
        "DeactivateNode": grpc.unary_unary_rpc_method_handler(
            servicer.unary_unary,
            request_deserializer=DeactivateNodeRequest.FromString,
            response_serializer=DeactivateNodeResponse.SerializeToString,
        ),
        "UnregisterNode": grpc.unary_unary_rpc_method_handler(
            servicer.unary_unary,
            request_deserializer=UnregisterNodeFleetRequest.FromString,
            response_serializer=UnregisterNodeFleetResponse.SerializeToString,
        ),
        "PullMessages": grpc.unary_unary_rpc_method_handler(
            servicer.unary_unary,
            request_deserializer=PullMessagesRequest.FromString,
            response_serializer=PullMessagesResponse.SerializeToString,
        ),
        "PushMessages": grpc.unary_unary_rpc_method_handler(
            servicer.unary_unary,
            request_deserializer=PushMessagesRequest.FromString,
            response_serializer=PushMessagesResponse.SerializeToString,
        ),
        "GetRun": grpc.unary_unary_rpc_method_handler(
            servicer.unary_unary,
            request_deserializer=GetRunRequest.FromString,
            response_serializer=GetRunResponse.SerializeToString,
        ),
        "SendNodeHeartbeat": grpc.unary_unary_rpc_method_handler(
            servicer.unary_unary,
            request_deserializer=SendNodeHeartbeatRequest.FromString,
            response_serializer=SendNodeHeartbeatResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "flwr.proto.Fleet", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


def _receive(conn: Any) -> None:
    _, receive, _, _, _, _, _, _ = conn
    receive()


def _send(conn: Any) -> None:
    _, receive, send, _, _, _, _, _ = conn
    receive()
    send(Message(RecordDict(), dst_node_id=0, message_type="query"))


def _get_run(conn: Any) -> None:
    _, _, _, get_run, _, _, _, _ = conn
    get_run(0)


class TestAuthenticateClientInterceptor(unittest.TestCase):
    """Test for client interceptor SuperNode authentication."""

    def setUp(self) -> None:
        """Initialize mock server and client."""
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=(("grpc.so_reuseport", int(False)),),
        )
        self._servicer = _MockServicer()
        _add_generic_handler(self._servicer, self._server)
        port = self._server.add_insecure_port("[::]:0")
        self._server.start()
        self._client_private_key, self._client_public_key = generate_key_pairs()

        self._connection = grpc_request_response
        self._address = f"localhost:{port}"

    @parameterized.expand([(_receive,), (_send,), (_get_run,)])  # type: ignore
    def test_client_auth_rpc(self, grpc_call: Callable[[Any], None]) -> None:
        """Test SuperNode authentication during create node."""
        # Execute
        with self._connection(
            self._address,
            True,
            Mock(invoke=lambda fn, *args, **kwargs: fn(*args, **kwargs)),
            GRPC_MAX_MESSAGE_LENGTH,
            None,
            (self._client_private_key, self._client_public_key),
        ) as conn:
            grpc_call(conn)

            received_metadata = self._servicer.received_client_metadata()
            assert received_metadata is not None

            metadata_dict = dict(received_metadata)
            actual_public_key = metadata_dict[PUBLIC_KEY_HEADER]
            signature = metadata_dict[SIGNATURE_HEADER]
            timestamp = metadata_dict[TIMESTAMP_HEADER]

            expected_public_key = public_key_to_bytes(self._client_public_key)

            # Assert
            assert metadata_dict[FLWR_COMPONENT_NAME_METADATA_KEY] == "SuperNode"
            assert FLWR_PACKAGE_NAME_METADATA_KEY in metadata_dict
            assert FLWR_PACKAGE_VERSION_METADATA_KEY in metadata_dict
            assert isinstance(signature, bytes)
            assert isinstance(timestamp, str)
            assert actual_public_key == expected_public_key
            assert verify_signature(
                self._client_public_key, timestamp.encode("ascii"), signature
            )
