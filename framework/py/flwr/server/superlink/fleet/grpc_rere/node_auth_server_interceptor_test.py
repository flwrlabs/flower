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
"""Flower server interceptor tests."""


import datetime
import hashlib
import unittest
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import grpc
from parameterized import parameterized

from flwr.common import now
from flwr.common.constant import (
    FLEET_API_GRPC_RERE_DEFAULT_ADDRESS,
    NOOP_ACCOUNT_NAME,
    NOOP_FLWR_AID,
    PUBLIC_KEY_HEADER,
    SIGNATURE_HEADER,
    SUPERLINK_NODE_ID,
    SYSTEM_TIME_TOLERANCE,
    TIMESTAMP_HEADER,
    TIMESTAMP_TOLERANCE,
    Status,
)
from flwr.common.typing import Fab, RunStatus
from flwr.proto.fab_pb2 import GetFabRequest, GetFabResponse  # pylint: disable=E0611
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
from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
from flwr.server.app import _run_fleet_api_grpc_rere
from flwr.server.superlink.linkstate.linkstate_factory import LinkStateFactory
from flwr.server.superlink.linkstate.linkstate_test import create_res_message
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME, NOOP_FEDERATION, RunType
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.primitives.asymmetric import (
    generate_key_pairs,
    public_key_to_bytes,
    sign_message,
)
from flwr.superlink.federation import NoOpFederationManager

from .node_auth_server_interceptor import NodeAuthServerInterceptor


class TestNodeAuthServerInterceptor(unittest.TestCase):  # pylint: disable=R0902
    """Node authentication server interceptor tests with node auth disabled."""

    enable_node_auth = False

    def setUp(self) -> None:
        """Initialize mock stub and server interceptor."""
        self.node_sk, self.node_pk = generate_key_pairs()
        self.node_pk_bytes = public_key_to_bytes(self.node_pk)

        objectstore_factory = ObjectStoreFactory()
        state_factory = LinkStateFactory(
            FLWR_IN_MEMORY_DB_NAME, NoOpFederationManager(), objectstore_factory
        )
        self.state = state_factory.state()
        self.store = objectstore_factory.store()

        self._server_interceptor = NodeAuthServerInterceptor(state_factory)
        self._server: grpc.Server = _run_fleet_api_grpc_rere(
            FLEET_API_GRPC_RERE_DEFAULT_ADDRESS,
            state_factory,
            objectstore_factory,
            self.enable_node_auth,
            None,
            [self._server_interceptor],
        )

        self._channel = grpc.insecure_channel("localhost:9092")
        self._register_node = self._channel.unary_unary(
            "/flwr.proto.Fleet/RegisterNode",
            request_serializer=RegisterNodeFleetRequest.SerializeToString,
            response_deserializer=RegisterNodeFleetResponse.FromString,
        )
        self._activate_node = self._channel.unary_unary(
            "/flwr.proto.Fleet/ActivateNode",
            request_serializer=ActivateNodeRequest.SerializeToString,
            response_deserializer=ActivateNodeResponse.FromString,
        )
        self._deactivate_node = self._channel.unary_unary(
            "/flwr.proto.Fleet/DeactivateNode",
            request_serializer=DeactivateNodeRequest.SerializeToString,
            response_deserializer=DeactivateNodeResponse.FromString,
        )
        self._unregister_node = self._channel.unary_unary(
            "/flwr.proto.Fleet/UnregisterNode",
            request_serializer=UnregisterNodeFleetRequest.SerializeToString,
            response_deserializer=UnregisterNodeFleetResponse.FromString,
        )
        self._pull_messages = self._channel.unary_unary(
            "/flwr.proto.Fleet/PullMessages",
            request_serializer=PullMessagesRequest.SerializeToString,
            response_deserializer=PullMessagesResponse.FromString,
        )
        self._push_messages = self._channel.unary_unary(
            "/flwr.proto.Fleet/PushMessages",
            request_serializer=PushMessagesRequest.SerializeToString,
            response_deserializer=PushMessagesResponse.FromString,
        )
        self._pull_object = self._channel.unary_unary(
            "/flwr.proto.Fleet/PullObject",
            request_serializer=PullObjectRequest.SerializeToString,
            response_deserializer=PullObjectResponse.FromString,
        )
        self._push_object = self._channel.unary_unary(
            "/flwr.proto.Fleet/PushObject",
            request_serializer=PushObjectRequest.SerializeToString,
            response_deserializer=PushObjectResponse.FromString,
        )
        self._get_run = self._channel.unary_unary(
            "/flwr.proto.Fleet/GetRun",
            request_serializer=GetRunRequest.SerializeToString,
            response_deserializer=GetRunResponse.FromString,
        )
        self._send_node_heartbeat = self._channel.unary_unary(
            "/flwr.proto.Fleet/SendNodeHeartbeat",
            request_serializer=SendNodeHeartbeatRequest.SerializeToString,
            response_deserializer=SendNodeHeartbeatResponse.FromString,
        )
        self._get_fab = self._channel.unary_unary(
            "/flwr.proto.Fleet/GetFab",
            request_serializer=GetFabRequest.SerializeToString,
            response_deserializer=GetFabResponse.FromString,
        )

    def tearDown(self) -> None:
        """Clean up grpc server."""
        self._server.stop(None)

    def _make_metadata(self) -> list[Any]:
        """Create metadata with signature and timestamp."""
        timestamp = now().isoformat()
        signature = sign_message(self.node_sk, timestamp.encode("ascii"))
        return [
            (PUBLIC_KEY_HEADER, self.node_pk_bytes),
            (SIGNATURE_HEADER, signature),
            (TIMESTAMP_HEADER, timestamp),
        ]

    def _make_metadata_with_invalid_signature(self) -> list[Any]:
        """Create metadata with invalid signature."""
        timestamp = now().isoformat()
        sk, _ = generate_key_pairs()
        signature = sign_message(sk, timestamp.encode("ascii"))
        return [
            (PUBLIC_KEY_HEADER, self.node_pk_bytes),
            (SIGNATURE_HEADER, signature),
            (TIMESTAMP_HEADER, timestamp),
        ]

    def _make_metadata_with_invalid_public_key(self) -> list[Any]:
        """Create metadata with invalid public key."""
        timestamp = now().isoformat()
        signature = sign_message(self.node_sk, timestamp.encode("ascii"))
        _, pk = generate_key_pairs()
        return [
            (PUBLIC_KEY_HEADER, public_key_to_bytes(pk)),
            (SIGNATURE_HEADER, signature),
            (TIMESTAMP_HEADER, timestamp),
        ]

    def _make_metadata_with_invalid_timestamp(self) -> list[Any]:
        """Create metadata with invalid timestamp."""
        timestamp = (
            now()
            - datetime.timedelta(seconds=TIMESTAMP_TOLERANCE + SYSTEM_TIME_TOLERANCE)
        ).isoformat()
        signature = sign_message(self.node_sk, timestamp.encode("ascii"))
        return [
            (PUBLIC_KEY_HEADER, self.node_pk_bytes),
            (SIGNATURE_HEADER, signature),
            (TIMESTAMP_HEADER, timestamp),
        ]

    def _test_register_node(self, metadata: list[Any]) -> Any:
        """Test RegisterNode."""
        return self._register_node.with_call(
            request=RegisterNodeFleetRequest(public_key=self.node_pk_bytes),
            metadata=metadata,
        )

    def _test_activate_node(self, metadata: list[Any]) -> Any:
        """Test ActivateNode."""
        self._create_node_in_linkstate(activate=False)
        req = ActivateNodeRequest(public_key=self.node_pk_bytes, heartbeat_interval=30)
        return self._activate_node.with_call(request=req, metadata=metadata)

    def _test_deactivate_node(self, metadata: list[Any]) -> Any:
        """Test DeactivateNode."""
        node_id = self._create_node_in_linkstate()
        req = DeactivateNodeRequest(node_id=node_id)
        return self._deactivate_node.with_call(request=req, metadata=metadata)

    def _test_unregister_node(self, metadata: list[Any]) -> Any:
        """Test UnregisterNode."""
        node_id = self._create_node_in_linkstate()
        req = UnregisterNodeFleetRequest(node_id=node_id)
        return self._unregister_node.with_call(request=req, metadata=metadata)

    def _test_pull_messages(self, metadata: list[Any]) -> Any:
        """Test PullMessages."""
        node_id = self._create_node_in_linkstate()
        req = PullMessagesRequest(node=Node(node_id=node_id))
        return self._pull_messages.with_call(request=req, metadata=metadata)

    def _create_dummy_run(
        self, running: bool = True, fab_hash: str | None = None
    ) -> int:
        """Create a dummy run in linkstate and return the run_id."""
        run_id = self.state.create_run(
            "",
            "",
            fab_hash,
            {},
            NOOP_FEDERATION,
            None,
            "",
            RunType.SERVER_APP,
        )
        if running:
            self.state.update_run_status(run_id, RunStatus(Status.STARTING, "", ""))
            self.state.update_run_status(run_id, RunStatus(Status.RUNNING, "", ""))
        return run_id

    def _test_push_messages(self, metadata: list[Any]) -> Any:
        """Test PushMessages."""
        node_id = self._create_node_in_linkstate()
        run_id = self._create_dummy_run()
        msg_proto = create_res_message(
            src_node_id=node_id, dst_node_id=SUPERLINK_NODE_ID, run_id=run_id
        )
        req = PushMessagesRequest(node=Node(node_id=node_id), messages_list=[msg_proto])
        return self._push_messages.with_call(request=req, metadata=metadata)

    def _test_pull_object(self, metadata: list[Any]) -> Any:
        """Test PullObject."""
        node_id = self._create_node_in_linkstate()
        run_id = self._create_dummy_run()
        req = PullObjectRequest(
            node=Node(node_id=node_id), run_id=run_id, object_id="1234"
        )
        # Mock store_traffic to avoid validation error when object_content is empty
        # This is because the object has been preregistered but not yet pushed
        with patch.object(self.state, "store_traffic"):
            return self._pull_object.with_call(request=req, metadata=metadata)

    def _test_push_object(self, metadata: list[Any]) -> Any:
        """Test PushObject."""
        node_id = self._create_node_in_linkstate()
        run_id = self._create_dummy_run()
        req = PushObjectRequest(
            node=Node(node_id=node_id),
            run_id=run_id,
            object_id="1234",
            object_content=b"1234",
        )
        # Mock store_traffic to avoid validation error when object_content is empty
        # This is because the object has been preregistered but not yet pushed
        with patch.object(self.state, "store_traffic"):
            return self._push_object.with_call(request=req, metadata=metadata)

    def _test_get_run(self, metadata: list[Any]) -> Any:
        """Test GetRun."""
        node_id = self._create_node_in_linkstate()
        run_id = self._create_dummy_run()
        req = GetRunRequest(node=Node(node_id=node_id), run_id=run_id)
        return self._get_run.with_call(request=req, metadata=metadata)

    def _test_send_node_heartbeat(self, metadata: list[Any]) -> Any:
        """Test SendNodeHeartbeat."""
        node_id = self._create_node_in_linkstate()
        req = SendNodeHeartbeatRequest(
            node=Node(node_id=node_id), heartbeat_interval=30.0
        )
        return self._send_node_heartbeat.with_call(request=req, metadata=metadata)

    def _test_get_fab(self, metadata: list[Any]) -> Any:
        """Test GetFab."""
        fab_content = b"mock fab content"
        fab_hash = self.state.store_fab(
            Fab(hashlib.sha256(fab_content).hexdigest(), fab_content, {})
        )
        node_id = self._create_node_in_linkstate()
        run_id = self._create_dummy_run(fab_hash=fab_hash)
        req = GetFabRequest(
            node=Node(node_id=node_id),
            run_id=run_id,
            hash_str=fab_hash,
        )
        return self._get_fab.with_call(request=req, metadata=metadata)

    def _create_node_in_linkstate(self, activate: bool = True) -> int:
        pk_bytes = self.node_pk_bytes
        node_id = self.state.create_node(
            owner_aid=NOOP_FLWR_AID,
            owner_name=NOOP_ACCOUNT_NAME,
            public_key=pk_bytes,
            heartbeat_interval=30,
        )
        if activate:
            self.state.activate_node(node_id, 30)
        return node_id

    rpcs = [
        (_test_register_node,),
        (_test_activate_node,),
        (_test_deactivate_node,),
        (_test_unregister_node,),
        (_test_pull_messages,),
        (_test_push_messages,),
        (_test_pull_object,),
        (_test_push_object,),
        (_test_get_run,),
        (_test_send_node_heartbeat,),
        (_test_get_fab,),
    ]

    @parameterized.expand(rpcs)  # type: ignore
    def test_successful_rpc_with_metadata(
        self, rpc: Callable[[Any, list[Any]], Any]
    ) -> None:
        """Test server interceptor for RPC."""
        # Skip registration and unregistration tests when node auth is enabled
        if self.enable_node_auth and rpc.__name__ in [
            self._test_register_node.__name__,
            self._test_unregister_node.__name__,
        ]:
            return

        # Execute
        _, call = rpc(self, self._make_metadata())

        # Assert
        assert call.code() == grpc.StatusCode.OK

    @parameterized.expand(rpcs)  # type: ignore
    def test_unsuccessful_rpc_with_invalid_signature(
        self, rpc: Callable[[Any, list[Any]], Any]
    ) -> None:
        """Test server interceptor for RPC unsuccessfully."""
        # Execute & Assert
        with self.assertRaises(grpc.RpcError) as cm:
            rpc(self, self._make_metadata_with_invalid_signature())
        assert cm.exception.code() == grpc.StatusCode.UNAUTHENTICATED

    @parameterized.expand(rpcs)  # type: ignore
    def test_unsuccessful_rpc_with_invalid_public_key(
        self, rpc: Callable[[Any, list[Any]], Any]
    ) -> None:
        """Test server interceptor for RPC unsuccessfully."""
        # Execute & Assert
        with self.assertRaises(grpc.RpcError) as cm:
            rpc(self, self._make_metadata_with_invalid_public_key())
        assert cm.exception.code() == grpc.StatusCode.UNAUTHENTICATED

    @parameterized.expand(rpcs)  # type: ignore
    def test_unsuccessful_rpc_with_invalid_timestamp(
        self, rpc: Callable[[Any, list[Any]], Any]
    ) -> None:
        """Test server interceptor for RPC unsuccessfully."""
        # Execute & Assert
        with self.assertRaises(grpc.RpcError) as cm:
            rpc(self, self._make_metadata_with_invalid_timestamp())
        assert cm.exception.code() == grpc.StatusCode.UNAUTHENTICATED


class TestNodeAuthServerInterceptorWithNodeAuthEnabled(TestNodeAuthServerInterceptor):
    """Node authentication server interceptor tests with node auth enabled."""

    enable_node_auth = True
