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
"""ServerAppIoServicer tests."""

# pylint: disable=too-many-lines


import hashlib
import os
import tempfile
import threading
import unittest
from datetime import timedelta
from unittest.mock import Mock, patch

import grpc
from parameterized import parameterized

from flwr.common import ConfigRecord, Context, Error, Message, RecordDict
from flwr.common.constant import (
    SERVERAPPIO_API_DEFAULT_SERVER_ADDRESS,
    SUPERLINK_NODE_ID,
    Status,
    SubStatus,
)
from flwr.common.message import get_message_to_descendant_id_mapping
from flwr.common.serde import context_to_proto, message_from_proto, run_status_to_proto
from flwr.common.serde_test import RecordMaker
from flwr.common.typing import Fab, RunStatus
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    ListAppsToLaunchRequest,
    ListAppsToLaunchResponse,
    PullAppInputsRequest,
    PullAppInputsResponse,
    PullAppMessagesRequest,
    PullAppMessagesResponse,
    PushAppMessagesRequest,
    PushAppMessagesResponse,
    PushAppOutputsRequest,
    PushAppOutputsResponse,
    RequestTokenRequest,
    RequestTokenResponse,
)
from flwr.proto.heartbeat_pb2 import (  # pylint: disable=E0611
    SendAppHeartbeatRequest,
    SendAppHeartbeatResponse,
)
from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    ConfirmMessageReceivedRequest,
    ConfirmMessageReceivedResponse,
)
from flwr.proto.message_pb2 import Message as ProtoMessage  # pylint: disable=E0611
from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    ObjectTree,
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import (  # pylint: disable=E0611
    UpdateRunStatusRequest,
    UpdateRunStatusResponse,
)
from flwr.proto.serverappio_pb2 import (  # pylint: disable=E0611
    GetNodesRequest,
    GetNodesResponse,
)
from flwr.server.superlink.linkstate.linkstate import LinkState
from flwr.server.superlink.linkstate.linkstate_factory import LinkStateFactory
from flwr.server.superlink.linkstate.linkstate_test import create_ins_message
from flwr.server.superlink.serverappio.serverappio_grpc import run_serverappio_api_grpc
from flwr.server.superlink.serverappio.serverappio_servicer import _raise_if
from flwr.server.superlink.utils import _STATUS_TO_MSG
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME, NOOP_FEDERATION, RunType
from flwr.supercore.date import now
from flwr.supercore.inflatable.inflatable_object import (
    get_all_nested_objects,
    get_object_id,
    get_object_tree,
    iterate_object_tree,
)
from flwr.supercore.interceptors import (
    APP_TOKEN_HEADER,
    AppIoTokenClientInterceptor,
    SuperExecAuthClientInterceptor,
)
from flwr.supercore.interceptors.superexec_auth_interceptor import (
    SERVERAPPIO_SUPEREXEC_METHODS,
)
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink.federation import NoOpFederationManager

# pylint: disable=broad-except,too-many-lines

_SUPEREXEC_SECRET = b"test-superexec-secret"


def test_raise_if_false() -> None:
    """."""
    # Prepare
    validation_error = False
    detail = "test"

    try:
        # Execute
        _raise_if(
            validation_error=validation_error,
            request_name="DummyRequest",
            detail=detail,
        )

        # Assert
        assert True
    except ValueError as err:
        raise AssertionError() from err
    except Exception as err:
        raise AssertionError() from err


def test_raise_if_true() -> None:
    """."""
    # Prepare
    validation_error = True
    detail = "test"

    try:
        # Execute
        _raise_if(
            validation_error=validation_error,
            request_name="DummyRequest",
            detail=detail,
        )

        # Assert
        raise AssertionError()
    except ValueError as err:
        assert str(err) == "Malformed DummyRequest: test"
    except Exception as err:
        raise AssertionError() from err


def _start_serverappio_with_port_retry(
    state_factory: LinkStateFactory,
    objectstore_factory: ObjectStoreFactory,
    start_port: int,
) -> grpc.Server:
    for offset in range(40):
        address = f"127.0.0.1:{start_port + offset}"
        try:
            return run_serverappio_api_grpc(
                address,
                state_factory,
                objectstore_factory,
                None,
                superexec_auth_secret=_SUPEREXEC_SECRET,
            )
        except RuntimeError as err:
            if "Failed to bind to address" in str(err):
                continue
            raise

    raise AssertionError(
        f"Could not bind ServerAppIo gRPC server starting at port {start_port}."
    )


def _create_shared_runtime(
    tmpdir: str,
) -> tuple[int, LinkState, grpc.Server, grpc.Server]:
    database_path = os.path.join(tmpdir, "shared.db")

    objectstore_factory_0 = ObjectStoreFactory()
    objectstore_factory_1 = ObjectStoreFactory()
    state_factory_0 = LinkStateFactory(
        database_path, NoOpFederationManager(), objectstore_factory_0
    )
    state_factory_1 = LinkStateFactory(
        database_path, NoOpFederationManager(), objectstore_factory_1
    )
    state_0 = state_factory_0.state()
    fab_content = b"mock fab content"
    fab_hash = state_0.store_fab(
        Fab(hashlib.sha256(fab_content).hexdigest(), fab_content, {})
    )

    run_id = state_0.create_run(
        "",
        "",
        fab_hash,
        {},
        NOOP_FEDERATION,
        None,
        "",
        RunType.SERVER_APP,
    )
    state_0.set_serverapp_context(
        run_id, Context(run_id, SUPERLINK_NODE_ID, {}, RecordDict(), {})
    )
    server_0 = _start_serverappio_with_port_retry(
        state_factory_0,
        objectstore_factory_0,
        start_port=19091,
    )
    server_1 = _start_serverappio_with_port_retry(
        state_factory_1,
        objectstore_factory_1,
        start_port=19141,
    )
    return run_id, state_0, server_0, server_1


def _request_token(channel: grpc.Channel, run_id: int) -> str:
    superexec_channel = grpc.intercept_channel(
        channel,
        SuperExecAuthClientInterceptor(
            master_secret=_SUPEREXEC_SECRET,
            protected_methods=SERVERAPPIO_SUPEREXEC_METHODS,
        ),
    )
    request_token = superexec_channel.unary_unary(
        "/flwr.proto.ServerAppIo/RequestToken",
        request_serializer=RequestTokenRequest.SerializeToString,
        response_deserializer=RequestTokenResponse.FromString,
    )
    token_response, token_call = request_token.with_call(
        RequestTokenRequest(run_id=run_id)
    )
    assert grpc.StatusCode.OK == token_call.code()
    token = str(token_response.token)
    assert token
    return token


def _claim_in_parallel(
    channel_0: grpc.Channel, channel_1: grpc.Channel, token: str
) -> list[grpc.StatusCode | None]:
    pull_app_inputs_0 = channel_0.unary_unary(
        "/flwr.proto.ServerAppIo/PullAppInputs",
        request_serializer=PullAppInputsRequest.SerializeToString,
        response_deserializer=PullAppInputsResponse.FromString,
    )
    pull_app_inputs_1 = channel_1.unary_unary(
        "/flwr.proto.ServerAppIo/PullAppInputs",
        request_serializer=PullAppInputsRequest.SerializeToString,
        response_deserializer=PullAppInputsResponse.FromString,
    )
    timeout = 5.0
    barrier = threading.Barrier(3)
    results: list[grpc.StatusCode | None] = [None, None]
    exceptions: list[Exception] = []

    def claim_inputs(idx: int, pull_fn: grpc.UnaryUnaryMultiCallable) -> None:
        try:
            barrier.wait(timeout=timeout)
            response, call = pull_fn.with_call(
                PullAppInputsRequest(token=token),
                metadata=((APP_TOKEN_HEADER, token),),
            )
            del response
            results[idx] = call.code()
        except grpc.RpcError as err:
            results[idx] = err.code()
        except Exception as ex:  # pylint: disable=broad-exception-caught
            exceptions.append(ex)

    threads = [
        threading.Thread(target=claim_inputs, args=(0, pull_app_inputs_0)),
        threading.Thread(target=claim_inputs, args=(1, pull_app_inputs_1)),
    ]
    for thread in threads:
        thread.start()
    try:
        barrier.wait(timeout=timeout)
    except threading.BrokenBarrierError as ex:
        exceptions.append(ex)
    for thread in threads:
        thread.join(timeout=timeout)

    alive_threads = [thread for thread in threads if thread.is_alive()]
    if alive_threads:
        raise AssertionError(
            f"Concurrent PullAppInputs test timed out; {len(alive_threads)} "
            f"thread(s) still alive after {timeout} seconds."
        )
    if exceptions:
        raise exceptions[0]
    return results


class TestServerAppIoServicer(unittest.TestCase):  # pylint: disable=R0902, R0904
    """ServerAppIoServicer tests for allowed RunStatuses."""

    def setUp(self) -> None:
        """Initialize mock stub and server interceptor."""
        objectstore_factory = ObjectStoreFactory()
        state_factory = LinkStateFactory(
            FLWR_IN_MEMORY_DB_NAME, NoOpFederationManager(), objectstore_factory
        )
        self.state = state_factory.state()
        self.store = objectstore_factory.store()
        self.node_pk = b"fake public key"
        self.node_id = self.state.create_node(
            "mock_owner", "fake_name", self.node_pk, 30
        )
        self.state.acknowledge_node_heartbeat(self.node_id, 1e3)

        self.status_to_msg = _STATUS_TO_MSG

        self._server: grpc.Server = run_serverappio_api_grpc(
            SERVERAPPIO_API_DEFAULT_SERVER_ADDRESS,
            state_factory,
            objectstore_factory,
            None,
            superexec_auth_secret=_SUPEREXEC_SECRET,
        )

        # Provide a valid metadata token on the default test channel so existing
        # servicer behavior tests continue to exercise business logic paths.
        self._auth_run_id = self.state.create_run(
            "", "", "", {}, NOOP_FEDERATION, None, "", RunType.SERVER_APP
        )
        auth_token = self.state.create_token(self._auth_run_id)
        assert auth_token is not None
        self._auth_token = auth_token
        _ = self.state.update_run_status(
            self._auth_run_id, RunStatus(Status.STARTING, "", "")
        )
        _ = self.state.update_run_status(
            self._auth_run_id, RunStatus(Status.RUNNING, "", "")
        )
        self._channel = grpc.intercept_channel(
            grpc.insecure_channel("localhost:9091"),
            AppIoTokenClientInterceptor(token=self._auth_token),
            SuperExecAuthClientInterceptor(
                master_secret=_SUPEREXEC_SECRET,
                protected_methods=SERVERAPPIO_SUPEREXEC_METHODS,
            ),
        )
        self._get_nodes = self._channel.unary_unary(
            "/flwr.proto.ServerAppIo/GetNodes",
            request_serializer=GetNodesRequest.SerializeToString,
            response_deserializer=GetNodesResponse.FromString,
        )
        self._push_messages = self._channel.unary_unary(
            "/flwr.proto.ServerAppIo/PushMessages",
            request_serializer=PushAppMessagesRequest.SerializeToString,
            response_deserializer=PushAppMessagesResponse.FromString,
        )
        self._pull_messages = self._channel.unary_unary(
            "/flwr.proto.ServerAppIo/PullMessages",
            request_serializer=PullAppMessagesRequest.SerializeToString,
            response_deserializer=PullAppMessagesResponse.FromString,
        )
        self._push_serverapp_outputs = self._channel.unary_unary(
            "/flwr.proto.ServerAppIo/PushAppOutputs",
            request_serializer=PushAppOutputsRequest.SerializeToString,
            response_deserializer=PushAppOutputsResponse.FromString,
        )
        self._update_run_status = self._channel.unary_unary(
            "/flwr.proto.ServerAppIo/UpdateRunStatus",
            request_serializer=UpdateRunStatusRequest.SerializeToString,
            response_deserializer=UpdateRunStatusResponse.FromString,
        )
        self._send_app_heartbeat = self._channel.unary_unary(
            "/flwr.proto.ServerAppIo/SendAppHeartbeat",
            request_serializer=SendAppHeartbeatRequest.SerializeToString,
            response_deserializer=SendAppHeartbeatResponse.FromString,
        )
        self._push_object = self._channel.unary_unary(
            "/flwr.proto.ServerAppIo/PushObject",
            request_serializer=PushObjectRequest.SerializeToString,
            response_deserializer=PushObjectResponse.FromString,
        )
        self._pull_object = self._channel.unary_unary(
            "/flwr.proto.ServerAppIo/PullObject",
            request_serializer=PullObjectRequest.SerializeToString,
            response_deserializer=PullObjectResponse.FromString,
        )
        self._confirm_message_received = self._channel.unary_unary(
            "/flwr.proto.ServerAppIo/ConfirmMessageReceived",
            request_serializer=ConfirmMessageReceivedRequest.SerializeToString,
            response_deserializer=ConfirmMessageReceivedResponse.FromString,
        )
        self._list_apps_to_launch = self._channel.unary_unary(
            "/flwr.proto.ServerAppIo/ListAppsToLaunch",
            request_serializer=ListAppsToLaunchRequest.SerializeToString,
            response_deserializer=ListAppsToLaunchResponse.FromString,
        )
        self._request_token = self._channel.unary_unary(
            "/flwr.proto.ServerAppIo/RequestToken",
            request_serializer=RequestTokenRequest.SerializeToString,
            response_deserializer=RequestTokenResponse.FromString,
        )
        self._pull_app_inputs = self._channel.unary_unary(
            "/flwr.proto.ServerAppIo/PullAppInputs",
            request_serializer=PullAppInputsRequest.SerializeToString,
            response_deserializer=PullAppInputsResponse.FromString,
        )

    def tearDown(self) -> None:
        """Clean up grpc server."""
        self._server.stop(None)

    def _transition_run_status(self, run_id: int, num_transitions: int) -> None:
        if num_transitions > 0:
            _ = self.state.update_run_status(run_id, RunStatus(Status.STARTING, "", ""))
        if num_transitions > 1:
            _ = self.state.update_run_status(run_id, RunStatus(Status.RUNNING, "", ""))
        if num_transitions > 2:
            _ = self.state.update_run_status(run_id, RunStatus(Status.FINISHED, "", ""))

    def _create_dummy_run(self, running: bool = True, *, fab_hash: str = "") -> int:
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
            self._transition_run_status(run_id, 2)
        return run_id

    def test_successful_get_node_if_running(self) -> None:
        """Test `GetNode` success."""
        # Prepare
        run_id = self._create_dummy_run()
        request = GetNodesRequest(run_id=run_id)

        # Execute
        response, call = self._get_nodes.with_call(request=request)

        # Assert
        assert isinstance(response, GetNodesResponse)
        assert grpc.StatusCode.OK == call.code()

    def _assert_get_nodes_not_allowed(self, run_id: int) -> None:
        """Assert `GetNodes` not allowed."""
        run_status = self.state.get_run_status({run_id})[run_id]
        request = GetNodesRequest(run_id=run_id)

        with self.assertRaises(grpc.RpcError) as e:
            self._get_nodes.with_call(request=request)
        assert e.exception.code() == grpc.StatusCode.PERMISSION_DENIED
        assert e.exception.details() == self.status_to_msg[run_status.status]

    @parameterized.expand(
        [
            (0,),  # Test not successful if RunStatus is pending.
            (1,),  # Test not successful if RunStatus is starting.
            (3,),  # Test not successful if RunStatus is finished.
        ]
    )  # type: ignore
    def test_get_nodes_not_successful_if_not_running(
        self, num_transitions: int
    ) -> None:
        """Test `GetNodes` not sucessful if RunStatus is pending."""
        # Prepare
        run_id = self._create_dummy_run(running=False)

        self._transition_run_status(run_id, num_transitions)

        # Execute & Assert
        self._assert_get_nodes_not_allowed(run_id)

    def test_successful_push_messages_if_running(self) -> None:
        """Test `PushMessages` success."""
        # Prepare
        run_id = self._create_dummy_run()
        message_ins = create_ins_message(
            src_node_id=SUPERLINK_NODE_ID, dst_node_id=self.node_id, run_id=run_id
        )

        # Construct message to descendant mapping
        message = message_from_proto(message_ins)
        descendant_mapping = get_message_to_descendant_id_mapping(message)
        request = PushAppMessagesRequest(
            messages_list=[message_ins],
            run_id=run_id,
            message_object_trees=[get_object_tree(message)],
        )

        # Execute
        response, call = self._push_messages.with_call(request=request)

        # Assert
        assert isinstance(response, PushAppMessagesResponse)
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

    def _assert_push_ins_messages_not_allowed(
        self, message: ProtoMessage, run_id: int
    ) -> None:
        """Assert `PushInsMessages` not allowed."""
        run_status = self.state.get_run_status({run_id})[run_id]
        request = PushAppMessagesRequest(messages_list=[message], run_id=run_id)

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
    def test_push_ins_messages_not_successful_if_not_running(
        self, num_transitions: int
    ) -> None:
        """Test `PushInsMessages` not successful if RunStatus is not running."""
        # Prepare
        run_id = self._create_dummy_run(running=False)
        message_ins = create_ins_message(
            src_node_id=SUPERLINK_NODE_ID, dst_node_id=self.node_id, run_id=run_id
        )

        self._transition_run_status(run_id, num_transitions)

        # Execute & Assert
        self._assert_push_ins_messages_not_allowed(message_ins, run_id)

    @parameterized.expand(
        [
            # The normal case:
            # The message is recognized by both `LinkState` and `ObjectStore`
            (True,),
            # The failure case:
            # The message is found in `LinkState` but not in `ObjectStore`
            (False,),
        ]
    )  # type: ignore
    def test_pull_messages_if_running(self, register_in_store: bool) -> None:
        """Test `PullMessages` success if objects are registered in ObjectStore."""
        # Prepare
        run_id = self._create_dummy_run()

        # Push Messages and reply
        message_ins = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=self.node_id, run_id=run_id
            )
        )
        # pylint: disable-next=W0212
        message_ins.metadata._message_id = message_ins.object_id  # type: ignore
        msg_id = self.state.store_message_ins(message=message_ins)
        msg_ = self.state.get_message_ins(node_id=self.node_id, limit=1)[0]

        reply_msg = Message(RecordDict(), reply_to=msg_)
        # pylint: disable-next=W0212
        reply_msg.metadata._message_id = reply_msg.object_id  # type: ignore
        self.state.store_message_res(message=reply_msg)

        # Register response in ObjectStore (so pulling message request can be completed)
        obj_ids_registered: list[str] = []
        if register_in_store:
            obj_ids_registered = self.store.preregister(
                run_id, get_object_tree(reply_msg)
            )

        request = PullAppMessagesRequest(message_ids=[str(msg_id)], run_id=run_id)

        # Execute
        response, call = self._pull_messages.with_call(request=request)

        # Assert
        assert isinstance(response, PullAppMessagesResponse)
        assert call.code() == grpc.StatusCode.OK

        if register_in_store:
            object_tree = response.message_object_trees[0]
            object_ids_in_response = [
                tree.object_id for tree in iterate_object_tree(object_tree)
            ]
            # Assert expected object_ids
            assert set(obj_ids_registered) == set(object_ids_in_response)
            # Assert the root node of the object tree is the message
            assert reply_msg.object_id == object_tree.object_id
        else:
            assert len(response.messages_list) == 0
            assert len(response.message_object_trees) == 0
            # Ins message was deleted
            assert self.state.num_message_ins() == 0

    @parameterized.expand(
        [
            # Reply with Message
            (RecordDict(), None),
            # Reply with Error
            (None, Error(code=0)),
        ]
    )  # type: ignore
    def test_successful_pull_messages_deletes_messages_in_linkstate(
        self, content: RecordDict | None, error: Error | None
    ) -> None:
        """Test `PullMessages` deletes messages from LinkState."""
        # Prepare
        run_id = self._create_dummy_run()

        # Push Messages and reply
        message_ins = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=self.node_id, run_id=run_id
            )
        )
        # pylint: disable-next=W0212
        message_ins.metadata._message_id = message_ins.object_id  # type: ignore

        msg_id = self.state.store_message_ins(message=message_ins)
        msg_ = self.state.get_message_ins(node_id=self.node_id, limit=1)[0]

        if content is not None:
            reply_msg = Message(content, reply_to=msg_)
        else:
            assert error is not None
            reply_msg = Message(error, reply_to=msg_)

        # pylint: disable-next=W0212
        reply_msg.metadata._message_id = reply_msg.object_id  # type: ignore

        self.state.store_message_res(message=reply_msg)
        # Register response in ObjectStore (so pulling message request can be completed)
        self.store.preregister(run_id, get_object_tree(reply_msg))
        request = PullAppMessagesRequest(message_ids=[str(msg_id)], run_id=run_id)

        # Execute
        response, call = self._pull_messages.with_call(request=request)

        # Assert
        assert isinstance(response, PullAppMessagesResponse)
        assert grpc.StatusCode.OK == call.code()
        assert self.state.num_message_ins() == 0
        assert self.state.num_message_res() == 0

    def _assert_pull_messages_not_allowed(self, run_id: int) -> None:
        """Assert `PullMessages` not allowed."""
        run_status = self.state.get_run_status({run_id})[run_id]
        request = PullAppMessagesRequest(run_id=run_id)

        with self.assertRaises(grpc.RpcError) as e:
            self._pull_messages.with_call(request=request)
        assert e.exception.code() == grpc.StatusCode.PERMISSION_DENIED
        assert e.exception.details() == self.status_to_msg[run_status.status]

    @parameterized.expand(
        [
            (0,),  # Test not successful if RunStatus is pending.
            (1,),  # Test not successful if RunStatus is starting.
            (3,),  # Test not successful if RunStatus is finished.
        ]
    )  # type: ignore
    def test_pull_messages_not_successful_if_not_running(
        self, num_transitions: int
    ) -> None:
        """Test `PullMessages` not successful if RunStatus is not running."""
        # Prepare
        run_id = self._create_dummy_run(running=False)

        self._transition_run_status(run_id, num_transitions)

        # Execute & Assert
        self._assert_pull_messages_not_allowed(run_id)

    def test_pull_message_from_expired_message_error(self) -> None:
        """Test that the servicer correctly handles the registration in the ObjectStore
        of an Error message created by the LinkState due to an expired TTL."""
        # Prepare
        run_id = self._create_dummy_run()

        # Push Messages and reply
        message_ins = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=self.node_id, run_id=run_id
            )
        )
        msg_id = self.state.store_message_ins(message=message_ins)

        # Simulate situation where the message has expired in the LinkState
        # This will trigger the creation of an Error message
        future_dt = now() + timedelta(seconds=message_ins.metadata.ttl + 0.1)
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = future_dt  # over TTL limit

            token = self.state.create_token(run_id)
            assert token is not None
            request = PullAppMessagesRequest(message_ids=[str(msg_id)], run_id=run_id)
            pull_messages_plain = grpc.insecure_channel("localhost:9091").unary_unary(
                "/flwr.proto.ServerAppIo/PullMessages",
                request_serializer=PullAppMessagesRequest.SerializeToString,
                response_deserializer=PullAppMessagesResponse.FromString,
            )

            # Execute
            response, call = pull_messages_plain.with_call(
                request=request,
                metadata=((APP_TOKEN_HEADER, token),),
            )

            # Assert
            assert isinstance(response, PullAppMessagesResponse)
            assert grpc.StatusCode.OK == call.code()

            # Assert that objects to pull points to a message carrying an error
            msg_res = message_from_proto(response.messages_list[0])
            assert msg_res.has_error()
            object_tree = response.message_object_trees[0]
            object_ids_in_response = [
                tree.object_id for tree in iterate_object_tree(object_tree)
            ]
            # expected a single object id (that of the error message)
            assert list(object_ids_in_response) == [msg_res.object_id]

    def test_push_serverapp_outputs_successful_if_running(self) -> None:
        """Test `PushServerAppOutputs` success."""
        # Prepare
        run_id = self._create_dummy_run(running=False)
        token = self.state.create_token(run_id)
        assert token is not None

        maker = RecordMaker()
        context = Context(
            run_id=run_id,
            node_id=0,
            node_config=maker.user_config(),
            state=maker.recorddict(1, 1, 1),
            run_config=maker.user_config(),
        )

        # Transition status to running. PushAppOutputsRequest is only
        # allowed in running status.
        self._transition_run_status(run_id, 2)
        request = PushAppOutputsRequest(
            token=token, run_id=run_id, context=context_to_proto(context)
        )

        # Execute
        response, call = self._push_serverapp_outputs.with_call(request=request)

        # Assert
        assert isinstance(response, PushAppOutputsResponse)
        assert grpc.StatusCode.OK == call.code()

    def _assert_push_serverapp_outputs_not_allowed(
        self, token: str, context: Context
    ) -> None:
        """Assert `PushServerAppOutputs` not allowed."""
        run_id = self.state.get_run_id_by_token(token)
        assert run_id is not None, "Invalid token is provided."
        run_status = self.state.get_run_status({run_id})[run_id]
        request = PushAppOutputsRequest(
            token=token, run_id=run_id, context=context_to_proto(context)
        )

        with self.assertRaises(grpc.RpcError) as e:
            self._push_serverapp_outputs.with_call(request=request)
        assert e.exception.code() == grpc.StatusCode.PERMISSION_DENIED
        assert e.exception.details() == self.status_to_msg[run_status.status]

    @parameterized.expand(
        [
            (0,),  # Test not successful if RunStatus is pending.
            (1,),  # Test not successful if RunStatus is starting.
            (3,),  # Test not successful if RunStatus is finished.
        ]
    )  # type: ignore
    def test_push_serverapp_outputs_not_successful_if_not_running(
        self, num_transitions: int
    ) -> None:
        """Test `PushServerAppOutputs` not successful if RunStatus is not running."""
        # Prepare
        run_id = self._create_dummy_run(running=False)
        token = self.state.create_token(run_id)
        assert token is not None

        maker = RecordMaker()
        context = Context(
            run_id=run_id,
            node_id=0,
            node_config=maker.user_config(),
            state=maker.recorddict(1, 1, 1),
            run_config=maker.user_config(),
        )

        self._transition_run_status(run_id, num_transitions)

        # Execute & Assert
        self._assert_push_serverapp_outputs_not_allowed(token, context)

    @parameterized.expand(
        [
            (0,),  # Test successful if RunStatus is pending.
            (1,),  # Test successful if RunStatus is starting.
            (2,),  # Test successful if RunStatus is running.
        ]
    )  # type: ignore
    def test_update_run_status_successful_if_not_finished(
        self, num_transitions: int
    ) -> None:
        """Test `UpdateRunStatus` success."""
        # Prepare
        run_id = self._create_dummy_run(running=False)
        _ = self.state.get_run_status({run_id})[run_id]
        next_run_status = RunStatus(Status.STARTING, "", "")

        if num_transitions > 0:
            _ = self.state.update_run_status(run_id, RunStatus(Status.STARTING, "", ""))
            next_run_status = RunStatus(Status.RUNNING, "", "")
        if num_transitions > 1:
            _ = self.state.update_run_status(run_id, RunStatus(Status.RUNNING, "", ""))
            next_run_status = RunStatus(Status.FINISHED, "", "")

        request = UpdateRunStatusRequest(
            run_id=run_id, run_status=run_status_to_proto(next_run_status)
        )

        # Execute
        response, call = self._update_run_status.with_call(request=request)

        # Assert
        assert isinstance(response, UpdateRunStatusResponse)
        assert grpc.StatusCode.OK == call.code()

    def test_update_run_status_not_successful_if_finished(self) -> None:
        """Test `UpdateRunStatus` not successful."""
        # Prepare
        run_id = self._create_dummy_run(running=False)
        _ = self.state.get_run_status({run_id})[run_id]
        _ = self.state.update_run_status(run_id, RunStatus(Status.FINISHED, "", ""))
        run_status = self.state.get_run_status({run_id})[run_id]
        next_run_status = RunStatus(Status.FINISHED, "", "")

        request = UpdateRunStatusRequest(
            run_id=run_id, run_status=run_status_to_proto(next_run_status)
        )

        with self.assertRaises(grpc.RpcError) as e:
            self._update_run_status.with_call(request=request)
        assert e.exception.code() == grpc.StatusCode.PERMISSION_DENIED
        assert e.exception.details() == self.status_to_msg[run_status.status]

    @parameterized.expand([(True,), (False,)])  # type: ignore
    def test_send_app_heartbeat(self, success: bool) -> None:
        """Test sending an app heartbeat."""
        # Prepare
        token = "test-token"
        request = SendAppHeartbeatRequest(token=token)
        mock_ack_method = Mock(return_value=success)
        self.state.acknowledge_app_heartbeat = mock_ack_method  # type: ignore

        # Execute
        response, _ = self._send_app_heartbeat.with_call(request=request)

        # Assert
        self.assertIsInstance(response, SendAppHeartbeatResponse)
        self.assertEqual(response.success, success)
        mock_ack_method.assert_called_once_with(token)

    def test_push_object_succesful(self) -> None:
        """Test `PushObject`."""
        # Prepare
        run_id = self._create_dummy_run()
        obj = ConfigRecord({"a": 123, "b": [4, 5, 6]})
        obj_b = obj.deflate()

        # Pre-register object
        self.store.preregister(run_id, get_object_tree(obj))

        # Execute
        req = PushObjectRequest(
            node=Node(node_id=SUPERLINK_NODE_ID),
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

        # Run is running but node ID isn't recognized
        self._transition_run_status(run_id, 2)
        req = PushObjectRequest(node=Node(node_id=123), run_id=run_id)
        with self.assertRaises(grpc.RpcError) as e:
            self._push_object(request=req)
        assert e.exception.code() == grpc.StatusCode.FAILED_PRECONDITION

        # Prepare
        obj = ConfigRecord({"a": 123, "b": [4, 5, 6]})
        obj_b = obj.deflate()

        # Push valid object but it hasn't been pre-registered
        req = PushObjectRequest(
            node=Node(node_id=SUPERLINK_NODE_ID),
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
            node=Node(node_id=SUPERLINK_NODE_ID),
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
        obj = ConfigRecord({"a": 123, "b": [4, 5, 6]})
        obj_b = obj.deflate()

        # Preregister object
        self.store.preregister(run_id, get_object_tree(obj))

        # Pull
        req = PullObjectRequest(
            node=Node(node_id=SUPERLINK_NODE_ID), run_id=run_id, object_id=obj.object_id
        )
        res: PullObjectResponse = self._pull_object(req)

        # Assert object content is b"" (it was never pushed)
        assert res.object_found
        assert not res.object_available
        assert res.object_content == b""

        # Put object in store, then check it can be pulled
        self.store.put(object_id=obj.object_id, object_content=obj_b)
        req = PullObjectRequest(
            node=Node(node_id=SUPERLINK_NODE_ID), run_id=run_id, object_id=obj.object_id
        )
        res = self._pull_object(req)

        # Assert, identical object pulled
        assert res.object_found
        assert res.object_available
        assert obj_b == res.object_content

    def test_pull_object_fails(self) -> None:
        """Test `PullObject` in unsuported scenarios."""
        run_id = self._create_dummy_run(running=False)
        # Run is not running
        req = PullObjectRequest(node=Node(node_id=123), run_id=run_id)
        with self.assertRaises(grpc.RpcError) as e:
            self._pull_object(request=req)
        assert e.exception.code() == grpc.StatusCode.PERMISSION_DENIED

        # Run is running but node ID isn't recognized
        self._transition_run_status(run_id, 2)
        req = PullObjectRequest(node=Node(node_id=123), run_id=run_id)
        with self.assertRaises(grpc.RpcError) as e:
            self._pull_object(request=req)
        assert e.exception.code() == grpc.StatusCode.FAILED_PRECONDITION

        # Attempt pulling object that doesn't exist
        req = PullObjectRequest(
            node=Node(node_id=SUPERLINK_NODE_ID), run_id=run_id, object_id="1234"
        )
        res: PullObjectResponse = self._pull_object(req)
        # Empty response
        assert not res.object_found

    def test_confirm_message_received_successful(self) -> None:
        """Test `ConfirmMessageReceived` success."""
        # Prepare
        run_id = self._create_dummy_run()
        proto = create_ins_message(
            src_node_id=SUPERLINK_NODE_ID, dst_node_id=self.node_id, run_id=run_id
        )
        message_ins = message_from_proto(proto)
        message_res = Message(
            RecordDict({"cfg": ConfigRecord({"key": "value"})}), reply_to=message_ins
        )

        # Prepare: Save reply message in ObjectStore
        all_objects = get_all_nested_objects(message_res)
        self.store.preregister(run_id, get_object_tree(message_res))
        for obj_id, obj in all_objects.items():
            self.store.put(object_id=obj_id, object_content=obj.deflate())

        # Assert: All objects are stored in the ObjectStore
        assert len(self.store) == len(all_objects)

        # Execute: Confirm message received
        request = ConfirmMessageReceivedRequest(
            node=Node(node_id=self.node_id),
            run_id=run_id,
            message_object_id=message_res.object_id,
        )
        response, call = self._confirm_message_received.with_call(request=request)

        # Assert
        assert isinstance(response, ConfirmMessageReceivedResponse)
        assert grpc.StatusCode.OK == call.code()

        # Assert: Message is removed from LinkState
        assert len(self.store) == 0

    def test_list_apps_to_launch(self) -> None:
        """Test `ListAppsToLaunch`."""
        # Prepare
        _run_id1 = self._create_dummy_run(running=True)  # Run ID 1 is running
        run_id2 = self._create_dummy_run(running=False)  # Run ID 2 is pending

        # Execute
        request = ListAppsToLaunchRequest()
        response, call = self._list_apps_to_launch.with_call(request=request)

        # Assert
        assert isinstance(response, ListAppsToLaunchResponse)
        assert grpc.StatusCode.OK == call.code()

        # Assert: Run ID 2 is returned
        assert response.run_ids == [run_id2]

    def test_request_token(self) -> None:
        """Test `RequestToken`."""
        # Prepare
        run_id = self._create_dummy_run(running=False)

        # Execute
        request = RequestTokenRequest(run_id=run_id)
        response1, call1 = self._request_token.with_call(request=request)
        response2, call2 = self._request_token.with_call(request=request)

        # Assert
        assert isinstance(response1, RequestTokenResponse)
        assert isinstance(response2, RequestTokenResponse)
        assert grpc.StatusCode.OK == call1.code()
        assert grpc.StatusCode.OK == call2.code()

        # Assert: Only one token is issued
        assert response1.token != ""
        assert response2.token == ""

    def test_request_token_fail_closed_for_finished_run(self) -> None:
        """Ensure `RequestToken` returns empty token for finished runs."""
        # Prepare
        run_id = self._create_dummy_run(running=False)
        self._transition_run_status(run_id, 2)
        assert self.state.update_run_status(
            run_id,
            RunStatus(Status.FINISHED, SubStatus.COMPLETED, "done"),
        )
        before = self.state.get_run_status({run_id})[run_id]

        # Execute
        response, call = self._request_token.with_call(
            request=RequestTokenRequest(run_id=run_id)
        )

        # Assert: token issuance fails closed
        assert isinstance(response, RequestTokenResponse)
        assert grpc.StatusCode.OK == call.code()
        assert response.token == ""

        # Assert: terminal run status/details remain unchanged
        after = self.state.get_run_status({run_id})[run_id]
        assert before.status == after.status == Status.FINISHED
        assert before.sub_status == after.sub_status == SubStatus.COMPLETED
        assert before.details == after.details == "done"

        # Assert: no token was left behind for this run
        token = self.state.create_token(run_id)
        assert token is not None
        self.state.delete_token(run_id)

    def test_run_status_transitions(self) -> None:
        """Test `RequestToken` and `PullAppInputs` transitions run status from PENDING
        to STARTING to RUNNING."""
        # Prepare: Create a run with FAB
        fab_content = b"mock fab content"
        fab_hash = self.state.store_fab(
            Fab(hashlib.sha256(fab_content).hexdigest(), fab_content, {})
        )
        run_id = self._create_dummy_run(running=False, fab_hash=fab_hash)

        # Set serverapp context
        context = Context(run_id, SUPERLINK_NODE_ID, {}, RecordDict(), {})
        self.state.set_serverapp_context(run_id, context)

        # Request token to transition to STARTING
        token_request = RequestTokenRequest(run_id=run_id)
        token_response, call = self._request_token.with_call(request=token_request)
        token = token_response.token

        # Assert: Response is successful and run status is STARTING
        assert isinstance(token_response, RequestTokenResponse)
        assert grpc.StatusCode.OK == call.code()
        run_status = self.state.get_run_status({run_id})[run_id]
        assert run_status.status == Status.STARTING

        # Execute: Pull app inputs
        request = PullAppInputsRequest(token=token)
        response, call = self._pull_app_inputs.with_call(request=request)

        # Assert: Response is successful and run status is now RUNNING
        assert isinstance(response, PullAppInputsResponse)
        assert grpc.StatusCode.OK == call.code()
        run_status = self.state.get_run_status({run_id})[run_id]
        assert run_status.status == Status.RUNNING


def test_ha_pull_app_inputs_claim_is_unique_across_replicas() -> None:
    """Ensure only one replica can claim STARTING -> RUNNING via PullAppInputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_id, state_0, server_0, server_1 = _create_shared_runtime(tmpdir)
        channel_0 = grpc.insecure_channel(server_0.bound_address)
        channel_1 = grpc.insecure_channel(server_1.bound_address)
        try:
            token = _request_token(channel_0, run_id)
            results = _claim_in_parallel(channel_0, channel_1, token)

            assert results.count(grpc.StatusCode.OK) == 1
            assert results.count(grpc.StatusCode.FAILED_PRECONDITION) == 1
            run_status = state_0.get_run_status({run_id})[run_id]
            assert run_status.status == Status.RUNNING
        finally:
            channel_0.close()
            channel_1.close()
            server_0.stop(None)
            server_1.stop(None)
