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
"""Flower FleetServicer tests."""


import hashlib
import unittest
from unittest.mock import Mock, patch

import grpc
from parameterized import parameterized

from flwr.common import ConfigRecord
from flwr.common.constant import (
    FLEET_API_GRPC_RERE_DEFAULT_ADDRESS,
    NOOP_ACCOUNT_NAME,
    NOOP_FLWR_AID,
    SUPERLINK_NODE_ID,
    Status,
)
from flwr.common.message import get_message_to_descendant_id_mapping
from flwr.common.serde import message_from_proto
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
from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    ConfirmMessageReceivedRequest,
    ConfirmMessageReceivedResponse,
    ObjectTree,
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
from flwr.server.app import _run_fleet_api_grpc_rere
from flwr.server.superlink.linkstate.linkstate_factory import LinkStateFactory
from flwr.server.superlink.linkstate.linkstate_test import (
    create_ins_message,
    create_res_message,
)
from flwr.server.superlink.utils import _STATUS_TO_MSG
from flwr.supercore.constant import (
    FLWR_IN_MEMORY_DB_NAME,
    NOOP_FEDERATION,
    NodeStatus,
    RunType,
)
from flwr.supercore.inflatable.inflatable_object import (
    get_all_nested_objects,
    get_object_id,
    get_object_tree,
    iterate_object_tree,
)
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink.federation import NoOpFederationManager


def create_fab(
    content: bytes = b"content", verifications: dict[str, str] | None = None
) -> Fab:
    """Create a FAB with a valid content hash."""
    return Fab(
        hash_str=hashlib.sha256(content).hexdigest(),
        content=content,
        verifications={"meta": "data"} if verifications is None else verifications,
    )


class TestFleetServicer(unittest.TestCase):  # pylint: disable=R0902, R0904
    """FleetServicer tests for allowed RunStatuses."""

    enable_node_auth = False

    def setUp(self) -> None:
        """Initialize mock stub and server interceptor."""
        objectstore_factory = ObjectStoreFactory()
        state_factory = LinkStateFactory(
            FLWR_IN_MEMORY_DB_NAME, NoOpFederationManager(), objectstore_factory
        )
        self.state = state_factory.state()
        self.store = objectstore_factory.store()
        self.node_pk = b"fake public key"

        self.status_to_msg = _STATUS_TO_MSG

        self._server: grpc.Server = _run_fleet_api_grpc_rere(
            FLEET_API_GRPC_RERE_DEFAULT_ADDRESS,
            state_factory,
            objectstore_factory,
            self.enable_node_auth,
            None,
            None,
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
        self._push_messages = self._channel.unary_unary(
            "/flwr.proto.Fleet/PushMessages",
            request_serializer=PushMessagesRequest.SerializeToString,
            response_deserializer=PushMessagesResponse.FromString,
        )
        self._pull_messages = self._channel.unary_unary(
            "/flwr.proto.Fleet/PullMessages",
            request_serializer=PullMessagesRequest.SerializeToString,
            response_deserializer=PullMessagesResponse.FromString,
        )
        self._get_run = self._channel.unary_unary(
            "/flwr.proto.Fleet/GetRun",
            request_serializer=GetRunRequest.SerializeToString,
            response_deserializer=GetRunResponse.FromString,
        )
        self._get_fab = self._channel.unary_unary(
            "/flwr.proto.Fleet/GetFab",
            request_serializer=GetFabRequest.SerializeToString,
            response_deserializer=GetFabResponse.FromString,
        )
        self._push_object = self._channel.unary_unary(
            "/flwr.proto.Fleet/PushObject",
            request_serializer=PushObjectRequest.SerializeToString,
            response_deserializer=PushObjectResponse.FromString,
        )
        self._pull_object = self._channel.unary_unary(
            "/flwr.proto.Fleet/PullObject",
            request_serializer=PullObjectRequest.SerializeToString,
            response_deserializer=PullObjectResponse.FromString,
        )
        self._confirm_message_received = self._channel.unary_unary(
            "/flwr.proto.Fleet/ConfirmMessageReceived",
            request_serializer=ConfirmMessageReceivedRequest.SerializeToString,
            response_deserializer=ConfirmMessageReceivedResponse.FromString,
        )

    def tearDown(self) -> None:
        """Clean up grpc server."""
        self._server.stop(None)

    def _create_dummy_node(self, activate: bool = True) -> int:
        """Create a dummy node."""
        node_id = self.state.create_node(
            NOOP_FLWR_AID, NOOP_ACCOUNT_NAME, self.node_pk, heartbeat_interval=30
        )
        if activate:
            self.state.acknowledge_node_heartbeat(node_id, heartbeat_interval=30)
        return node_id

    def _create_dummy_run(self, running: bool = True, fab_hash: str = "") -> int:
        """Create a dummy run."""
        run_id = self.state.create_run(
            fab_id="",
            fab_version="",
            fab_hash=fab_hash,
            override_config={},
            federation=NOOP_FEDERATION,
            federation_config=None,
            flwr_aid="",
            run_type=RunType.SERVER_APP,
        )
        if running:
            self._transition_run_status(run_id, 2)
        return run_id

    def _transition_run_status(self, run_id: int, num_transitions: int) -> None:
        if num_transitions > 0:
            _ = self.state.update_run_status(run_id, RunStatus(Status.STARTING, "", ""))
        if num_transitions > 1:
            _ = self.state.update_run_status(run_id, RunStatus(Status.RUNNING, "", ""))
        if num_transitions > 2:
            _ = self.state.update_run_status(run_id, RunStatus(Status.FINISHED, "", ""))

    def test_register_node_success(self) -> None:
        """Test `RegisterNode` success."""
        # Prepare
        public_key = b"test_register_public_key"
        request = RegisterNodeFleetRequest(public_key=public_key)

        # Execute: Registeration should be blocked when node authentication is enabled
        if self.enable_node_auth:
            with self.assertRaises(grpc.RpcError) as cm:
                self._register_node.with_call(request=request)
            assert cm.exception.code() == grpc.StatusCode.FAILED_PRECONDITION
            return

        # Execute: Allow registration when node authentication is disabled
        response, call = self._register_node.with_call(request=request)

        # Assert
        assert isinstance(response, RegisterNodeFleetResponse)
        assert grpc.StatusCode.OK == call.code()
        # Verify node was created in state
        node_id = self.state.get_node_id_by_public_key(public_key)
        assert node_id is not None
        assert node_id > 0
        assert response.node_id == node_id

    def test_activate_node_success(self) -> None:
        """Test `ActivateNode` success."""
        # Prepare: Register a node first
        public_key = b"test_activate_public_key"
        self.state.create_node(NOOP_FLWR_AID, NOOP_ACCOUNT_NAME, public_key, 0)
        request = ActivateNodeRequest(public_key=public_key, heartbeat_interval=30)

        # Execute
        response, call = self._activate_node.with_call(request=request)

        # Assert
        assert isinstance(response, ActivateNodeResponse)
        assert grpc.StatusCode.OK == call.code()
        assert response.node_id > 0
        # Verify node status is ONLINE
        node_info = self.state.get_node_info(node_ids=[response.node_id])[0]
        assert node_info.status == NodeStatus.ONLINE

    def test_activate_node_not_found(self) -> None:
        """Test `ActivateNode` with non-existent public key."""
        # Prepare
        public_key = b"non_existent_public_key"
        request = ActivateNodeRequest(public_key=public_key, heartbeat_interval=30)

        # Execute and assert
        with self.assertRaises(grpc.RpcError) as cm:
            self._activate_node.with_call(request=request)
        assert cm.exception.code() == grpc.StatusCode.FAILED_PRECONDITION

    def test_deactivate_node_success(self) -> None:
        """Test `DeactivateNode` success."""
        # Prepare: Create and activate a node
        public_key = b"test_deactivate_public_key"
        node_id = self.state.create_node(
            NOOP_FLWR_AID, NOOP_ACCOUNT_NAME, public_key, 30
        )
        self.state.activate_node(node_id, 30)
        request = DeactivateNodeRequest(node_id=node_id)

        # Execute
        response, call = self._deactivate_node.with_call(request=request)

        # Assert
        assert isinstance(response, DeactivateNodeResponse)
        assert grpc.StatusCode.OK == call.code()
        # Verify node status is OFFLINE
        node_info = self.state.get_node_info(node_ids=[node_id])[0]
        assert node_info.status == NodeStatus.OFFLINE

    def test_deactivate_node_failure(self) -> None:
        """Test `DeactivateNode` with invalid node_id."""
        # Prepare: Use a non-existent node_id
        request = DeactivateNodeRequest(node_id=99999)

        # Execute and assert
        with self.assertRaises(grpc.RpcError) as cm:
            self._deactivate_node.with_call(request=request)
        assert cm.exception.code() == grpc.StatusCode.FAILED_PRECONDITION

    def test_unregister_node_success(self) -> None:
        """Test `UnregisterNode` success."""
        # Prepare: Create a node
        public_key = b"test_unregister_public_key"
        node_id = self.state.create_node(
            NOOP_FLWR_AID, NOOP_ACCOUNT_NAME, public_key, 0
        )
        request = UnregisterNodeFleetRequest(node_id=node_id)

        # Execute: Unregistration should be blocked when node authentication is enabled
        if self.enable_node_auth:
            with self.assertRaises(grpc.RpcError) as cm:
                self._unregister_node.with_call(request=request)
            assert cm.exception.code() == grpc.StatusCode.FAILED_PRECONDITION
            return

        # Execute: Allow unregistration when node authentication is disabled
        response, call = self._unregister_node.with_call(request=request)

        # Assert
        assert isinstance(response, UnregisterNodeFleetResponse)
        assert grpc.StatusCode.OK == call.code()
        # Verify node status is UNREGISTERED
        node_info = self.state.get_node_info(node_ids=[node_id])[0]
        assert node_info.status == NodeStatus.UNREGISTERED

    def test_successful_push_messages_if_running(self) -> None:
        """Test `PushMessages` success."""
        # Prepare
        node_id = self._create_dummy_node()
        run_id = self._create_dummy_run()
        msg_proto = create_res_message(
            src_node_id=node_id, dst_node_id=SUPERLINK_NODE_ID, run_id=run_id
        )

        # Construct message to descendant mapping
        message = message_from_proto(msg_proto)
        descendant_mapping = get_message_to_descendant_id_mapping(message)

        request = PushMessagesRequest(
            node=Node(node_id=node_id),
            messages_list=[msg_proto],
            message_object_trees=[get_object_tree(message)],
        )

        # Execute
        response, call = self._push_messages.with_call(request=request)

        # Assert
        assert isinstance(response, PushMessagesResponse)
        assert grpc.StatusCode.OK == call.code()

        # Assert: check that response indicates all objects need pushing
        expected_object_ids = {message.object_id}  # message
        expected_object_ids |= {
            obj_id
            for obj_ids in descendant_mapping.values()
            for obj_id in obj_ids.object_ids
        }  # descendants
        # Construct a single set with all object ids
        requested_object_ids = set(response.objects_to_push)
        assert expected_object_ids == requested_object_ids

    def _assert_push_messages_not_allowed(self, node_id: int, run_id: int) -> None:
        """Assert `PushMessages` not allowed."""
        run_status = self.state.get_run_status({run_id})[run_id]

        msg_proto = create_res_message(
            src_node_id=node_id, dst_node_id=SUPERLINK_NODE_ID, run_id=run_id
        )
        request = PushMessagesRequest(
            node=Node(node_id=node_id), messages_list=[msg_proto]
        )

        with self.assertRaises(grpc.RpcError) as e:
            self._push_messages.with_call(request=request)
        assert e.exception.code() == grpc.StatusCode.PERMISSION_DENIED
        assert e.exception.details() == self.status_to_msg[run_status.status]

    @parameterized.expand(
        [
            (0,),  # Test not successful if RunStatus is pending.
            (1,),  # Test not successful if RunStatus is starting.
            (3,),  # Test not successful if RunStatus is finished.
        ]
    )  # type: ignore
    def test_push_messages_not_successful_if_not_running(
        self, num_transitions: int
    ) -> None:
        """Test `PushMessages` not successful if RunStatus is not running."""
        # Prepare
        node_id = self._create_dummy_node()
        run_id = self._create_dummy_run(running=False)
        self._transition_run_status(run_id, num_transitions)

        # Execute & Assert
        self._assert_push_messages_not_allowed(node_id, run_id)

    @parameterized.expand(
        [
            (True,),
            (False,),
        ]
    )  # type: ignore
    def test_pull_messages_if_running(
        self,
        register_in_store: bool,
    ) -> None:
        """Test `PullMessages` success if objects are registered in ObjectStore."""
        # Prepare
        node_id = self._create_dummy_node()

        run_id = self._create_dummy_run()

        # Let's insert a Message in the LinkState and register it in the ObjectStore
        message_ins = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id, run_id=run_id
            )
        )
        # pylint: disable-next=W0212
        message_ins.metadata._message_id = message_ins.object_id  # type: ignore
        self.state.store_message_ins(message=message_ins)
        obj_ids_registered: list[str] = []
        if register_in_store:
            obj_ids_registered = self.store.preregister(
                run_id, get_object_tree(message_ins)
            )

        request = PullMessagesRequest(node=Node(node_id=node_id))

        # Execute
        response, call = self._pull_messages.with_call(request=request)

        # Assert
        assert isinstance(response, PullMessagesResponse)
        assert call.code() == grpc.StatusCode.OK

        if register_in_store:
            object_tree = response.message_object_trees[0]
            object_ids_in_response = [
                tree.object_id for tree in iterate_object_tree(object_tree)
            ]
            # Assert expected object_ids
            assert set(obj_ids_registered) == set(object_ids_in_response)
            # Assert the root node of the object tree is the message
            assert message_ins.object_id == object_tree.object_id
        else:
            assert len(response.messages_list) == 0
            assert len(response.message_object_trees) == 0
            # Ins message was deleted
            assert self.state.num_message_ins() == 0

    def test_successful_get_run_if_running(self) -> None:
        """Test `GetRun` success."""
        # Prepare
        self._create_dummy_node()
        run_id = self._create_dummy_run()
        request = GetRunRequest(run_id=run_id)

        # Execute
        response, call = self._get_run.with_call(request=request)

        # Assert
        assert isinstance(response, GetRunResponse)
        assert grpc.StatusCode.OK == call.code()

    def _assert_get_run_not_allowed(self, run_id: int) -> None:
        """Assert `GetRun` not allowed."""
        run_status = self.state.get_run_status({run_id})[run_id]
        request = GetRunRequest(run_id=run_id)

        with self.assertRaises(grpc.RpcError) as e:
            self._get_run.with_call(request=request)
        assert e.exception.code() == grpc.StatusCode.PERMISSION_DENIED
        assert e.exception.details() == self.status_to_msg[run_status.status]

    @parameterized.expand(
        [
            (0,),  # Test not successful if RunStatus is pending.
            (1,),  # Test not successful if RunStatus is starting.
            (3,),  # Test not successful if RunStatus is finished.
        ]
    )  # type: ignore
    def test_get_run_not_successful_if_not_running(self, num_transitions: int) -> None:
        """Test `GetRun` not successful if RunStatus is not running."""
        # Prepare
        run_id = self._create_dummy_run(running=False)
        self._transition_run_status(run_id, num_transitions)

        # Execute & Assert
        self._assert_get_run_not_allowed(run_id)

    def test_get_run_permission_denied_if_node_not_in_federation(self) -> None:
        """Test `GetRun` raises PERMISSION_DENIED when node is not in federation."""
        # Prepare
        node_id = self._create_dummy_node()
        run_id = self._create_dummy_run()

        # Mock federation manager to exclude the node
        mock_has_node = Mock(return_value=False)
        self.state.federation_manager.has_node = mock_has_node  # type: ignore
        request = GetRunRequest(run_id=run_id, node=Node(node_id=node_id))

        # Execute and assert
        with self.assertRaises(grpc.RpcError) as e:
            self._get_run.with_call(request=request)

        assert e.exception.code() == grpc.StatusCode.PERMISSION_DENIED

    def test_successful_get_fab_if_running(self) -> None:
        """Test `GetFab` success."""
        # Prepare
        node_id = self._create_dummy_node()
        fab_content = b"content"
        fab_hash = self.state.store_fab(create_fab(content=fab_content))
        run_id = self._create_dummy_run(fab_hash=fab_hash)

        # Transition status to running. GetFab RPC is only allowed in running status.
        request = GetFabRequest(
            node=Node(node_id=node_id), hash_str=fab_hash, run_id=run_id
        )

        # Execute
        response, call = self._get_fab.with_call(request=request)

        # Assert
        assert isinstance(response, GetFabResponse)
        assert grpc.StatusCode.OK == call.code()

    def _assert_get_fab_not_allowed(
        self, node_id: int, hash_str: str, run_id: int
    ) -> None:
        """Assert `GetFab` not allowed."""
        run_status = self.state.get_run_status({run_id})[run_id]
        request = GetFabRequest(
            node=Node(node_id=node_id), hash_str=hash_str, run_id=run_id
        )

        with self.assertRaises(grpc.RpcError) as e:
            self._get_fab.with_call(request=request)
        assert e.exception.code() == grpc.StatusCode.PERMISSION_DENIED
        assert e.exception.details() == self.status_to_msg[run_status.status]

    @parameterized.expand(
        [
            (0,),  # Test not successful if RunStatus is pending.
            (1,),  # Test not successful if RunStatus is starting.
            (3,),  # Test not successful if RunStatus is finished.
        ]
    )  # type: ignore
    def test_get_fab_not_successful_if_not_running(self, num_transitions: int) -> None:
        """Test `GetFab` not successful if RunStatus is not running."""
        # Prepare
        node_id = self._create_dummy_node()
        fab_content = b"content"
        fab_hash = self.state.store_fab(create_fab(content=fab_content))
        run_id = self._create_dummy_run(running=False, fab_hash=fab_hash)

        self._transition_run_status(run_id, num_transitions)

        # Execute & Assert
        self._assert_get_fab_not_allowed(node_id, fab_hash, run_id)

    def test_get_fab_permission_denied_if_node_not_in_federation(self) -> None:
        """Test `GetFab` raises PERMISSION_DENIED when node is not in federation."""
        # Prepare
        node_id = self._create_dummy_node()
        fab_content = b"content"
        fab_hash = self.state.store_fab(create_fab(content=fab_content))
        run_id = self._create_dummy_run(fab_hash=fab_hash)

        # Mock federation manager to exclude the node
        mock_has_node = Mock(return_value=False)
        self.state.federation_manager.has_node = mock_has_node  # type: ignore

        request = GetFabRequest(
            node=Node(node_id=node_id), hash_str=fab_hash, run_id=run_id
        )

        # Execute and assert
        with self.assertRaises(grpc.RpcError) as e:
            self._get_fab.with_call(request=request)

        assert e.exception.code() == grpc.StatusCode.PERMISSION_DENIED

    def test_get_fab_permission_denied_if_hash_mismatches_run(self) -> None:
        """Test `GetFab` rejects hashes that do not match the run FAB hash."""
        node_id = self._create_dummy_node()
        fab_content = b"content"
        fab_hash = self.state.store_fab(create_fab(content=fab_content))
        run_id = self._create_dummy_run(fab_hash=fab_hash)
        wrong_hash = "0" * len(fab_hash)

        request = GetFabRequest(
            node=Node(node_id=node_id), hash_str=wrong_hash, run_id=run_id
        )

        with self.assertRaises(grpc.RpcError) as e:
            self._get_fab.with_call(request=request)

        assert e.exception.code() == grpc.StatusCode.PERMISSION_DENIED
        assert "does not match run FAB hash" in e.exception.details()

    def test_push_object_succesful(self) -> None:
        """Test `PushObject`."""
        # Prepare
        run_id = self._create_dummy_run()
        node_id = self._create_dummy_node()
        obj = ConfigRecord({"a": 123, "b": [4, 5, 6]})
        obj_b = obj.deflate()

        # Pre-register object
        self.store.preregister(run_id, get_object_tree(obj))

        # Execute
        req = PushObjectRequest(
            node=Node(node_id=node_id),
            run_id=run_id,
            object_id=obj.object_id,
            object_content=obj_b,
        )
        res: PushObjectResponse = self._push_object(request=req)

        # Empty response
        assert res.stored

    def test_push_object_fails(self) -> None:
        """Test `PushObject` in unsupported scenarios."""
        run_id = self._create_dummy_run(running=False)
        # Run is not running
        req = PushObjectRequest(node=Node(node_id=123), run_id=run_id)
        with self.assertRaises(grpc.RpcError) as e:
            self._push_object(request=req)
        assert e.exception.code() == grpc.StatusCode.PERMISSION_DENIED

        # Prepare
        self._transition_run_status(run_id, 2)
        node_id = self._create_dummy_node()
        obj = ConfigRecord({"a": 123, "b": [4, 5, 6]})
        obj_b = obj.deflate()

        # Push valid object but it hasn't been pre-registered
        req = PushObjectRequest(
            node=Node(node_id=node_id),
            run_id=run_id,
            object_id=obj.object_id,
            object_content=obj_b,
        )
        res: PushObjectResponse = self._push_object(request=req)

        # Assert: object not inserted
        assert not res.stored

        # Push valid object but its hash doesnt match the one passed in the request
        # Preregister under a different object-id
        fake_object_id = get_object_id(b"1234")
        self.store.preregister(run_id, ObjectTree(object_id=fake_object_id))

        # Execute
        req = PushObjectRequest(
            node=Node(node_id=node_id),
            run_id=run_id,
            object_id=fake_object_id,
            object_content=obj_b,
        )
        res = self._push_object(request=req)

        # Assert: object not inserted
        assert not res.stored

    def test_pull_object_successful(self) -> None:
        """Test `PullObject` functionality."""
        # Prepare
        run_id = self._create_dummy_run()
        node_id = self._create_dummy_node()
        obj = ConfigRecord({"a": 123, "b": [4, 5, 6]})
        obj_b = obj.deflate()

        # Preregister object
        self.store.preregister(run_id, get_object_tree(obj))

        # Mock store_traffic to avoid validation error when object_content is empty
        # This is because the object has been preregistered but not yet pushed
        with patch.object(self.state, "store_traffic"):
            # Pull
            req = PullObjectRequest(
                node=Node(node_id=node_id), run_id=run_id, object_id=obj.object_id
            )
            res: PullObjectResponse = self._pull_object(req)

            # Assert object content is b"" (it was never pushed)
            assert res.object_found
            assert not res.object_available
            assert res.object_content == b""

        # Put object in store, then check it can be pulled
        self.store.put(object_id=obj.object_id, object_content=obj_b)
        req = PullObjectRequest(
            node=Node(node_id=node_id), run_id=run_id, object_id=obj.object_id
        )
        res = self._pull_object(req)

        # Assert, identical object pulled
        assert res.object_found
        assert res.object_available
        assert obj_b == res.object_content

    def test_pull_object_fails(self) -> None:
        """Test `PullObject` in unsupported scenarios."""
        run_id = self._create_dummy_run(running=False)
        # Run is not running
        req = PullObjectRequest(node=Node(node_id=123), run_id=run_id)
        with self.assertRaises(grpc.RpcError) as e:
            self._pull_object(request=req)
        assert e.exception.code() == grpc.StatusCode.PERMISSION_DENIED

        # Attempt pulling object that doesn't exist
        self._transition_run_status(run_id, 2)
        node_id = self._create_dummy_node()
        # Mock store_traffic to avoid validation error when object_content is empty
        # This is because the object has been preregistered but not yet pushed
        with patch.object(self.state, "store_traffic"):
            req = PullObjectRequest(
                node=Node(node_id=node_id), run_id=run_id, object_id="1234"
            )
            res: PullObjectResponse = self._pull_object(req)

        # Empty response
        assert not res.object_found

    def test_confirm_message_received_successful(self) -> None:
        """Test `ConfirmMessageReceived` functionality."""
        # Prepare
        node_id = self._create_dummy_node()
        run_id = self._create_dummy_run()

        # Prepare: Create a message
        msg_proto = create_res_message(
            src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id, run_id=run_id
        )
        message = message_from_proto(msg_proto)
        # pylint: disable-next=E1137
        message.content["test_config"] = ConfigRecord({"a": 123, "b": [4, 5, 6]})
        message.metadata.__dict__["_message_id"] = message.object_id

        # Prepare: Store message in ObjectStore
        all_objects = get_all_nested_objects(message)
        self.store.preregister(run_id, get_object_tree(message))
        for obj_id, obj in all_objects.items():
            self.store.put(object_id=obj_id, object_content=obj.deflate())

        # Assert: Message is in ObjectStore
        assert len(self.store) == len(all_objects)

        # Execute: Confirm message received
        req = ConfirmMessageReceivedRequest(
            node=Node(node_id=node_id),
            run_id=run_id,
            message_object_id=message.object_id,
        )
        self._confirm_message_received(request=req)

        # Assert: Message is removed from ObjectStore
        assert len(self.store) == 0

    def test_push_object_records_traffic(self) -> None:
        """Test `PushObject` records traffic data."""
        # Prepare
        run_id = self._create_dummy_run()
        node_id = self._create_dummy_node()
        obj = ConfigRecord({"a": 321, "b": [6, 5, 4]})
        obj_b = obj.deflate()

        # Pre-register object
        self.store.preregister(run_id, get_object_tree(obj))

        # Get initial traffic
        run_before = self.state.get_run_info(run_ids=[run_id])[0]
        bytes_recv_before = run_before.bytes_recv

        # Execute
        req = PushObjectRequest(
            node=Node(node_id=node_id),
            run_id=run_id,
            object_id=obj.object_id,
            object_content=obj_b,
        )
        res: PushObjectResponse = self._push_object(request=req)

        # Assert
        assert res.stored
        run_after = self.state.get_run_info(run_ids=[run_id])[0]
        # Verify traffic was recorded
        assert run_after.bytes_recv == bytes_recv_before + len(obj_b)
        assert run_after.bytes_sent == 0  # No bytes sent during push

    def test_pull_object_records_traffic(self) -> None:
        """Test `PullObject` records traffic data."""
        # Prepare
        run_id = self._create_dummy_run()
        node_id = self._create_dummy_node()
        obj = ConfigRecord({"a": 789, "b": [6, 7, 8]})
        obj_b = obj.deflate()

        # Preregister and store object
        self.store.preregister(run_id, get_object_tree(obj))
        self.store.put(object_id=obj.object_id, object_content=obj_b)

        # Get initial traffic
        run_before = self.state.get_run_info(run_ids=[run_id])[0]
        bytes_sent_before = run_before.bytes_sent

        # Execute
        req = PullObjectRequest(
            node=Node(node_id=node_id), run_id=run_id, object_id=obj.object_id
        )
        res: PullObjectResponse = self._pull_object(req)

        # Assert
        assert res.object_found
        assert res.object_available
        run_after = self.state.get_run_info(run_ids=[run_id])[0]
        # Verify traffic was recorded
        assert run_after.bytes_sent == bytes_sent_before + len(obj_b)
        assert run_after.bytes_recv == 0  # No bytes received during pull


class TestFleetServicerWithNodeAuthEnabled(TestFleetServicer):
    """FleetServicer tests for allowed RunStatuses."""

    enable_node_auth = True
