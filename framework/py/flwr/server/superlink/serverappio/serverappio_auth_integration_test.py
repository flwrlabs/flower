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
"""ServerAppIo auth interceptor integration tests."""


import tempfile
import unittest
from collections.abc import Callable

import grpc
from google.protobuf.message import Message as GrpcMessage
from parameterized import parameterized

from flwr.common.constant import SERVERAPPIO_API_DEFAULT_SERVER_ADDRESS
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    PullAppMessagesRequest,
    PullAppMessagesResponse,
    PushAppMessagesRequest,
    PushAppMessagesResponse,
    SendTaskHeartbeatRequest,
    SendTaskHeartbeatResponse,
)
from flwr.proto.log_pb2 import (  # pylint: disable=E0611
    PushLogsRequest,
    PushLogsResponse,
)
from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    ConfirmMessageReceivedRequest,
    ConfirmMessageReceivedResponse,
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.serverappio_pb2 import (  # pylint: disable=E0611
    GetNodesRequest,
    GetNodesResponse,
)
from flwr.server.superlink.linkstate.linkstate_factory import LinkStateFactory
from flwr.server.superlink.serverappio.serverappio_grpc import run_serverappio_api_grpc
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME, NOOP_FEDERATION, RunType
from flwr.supercore.interceptors import (
    AUTHENTICATION_FAILED_MESSAGE,
    TASK_TOKEN_HEADER,
    AppIoTokenClientInterceptor,
    SuperExecAuthClientInterceptor,
)
from flwr.supercore.interceptors.superexec_auth_interceptor import (
    SERVERAPPIO_SUPEREXEC_METHODS,
)
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink.federation import NoOpFederationManager

_SUPEREXEC_SECRET = b"test-superexec-secret"


class TestServerAppIoAuthIntegration(unittest.TestCase):  # pylint: disable=R0902
    """Integration tests for ServerAppIo token-auth interceptor behavior."""

    def setUp(self) -> None:
        """Start the ServerAppIo gRPC API without client-side auth helpers."""
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.addCleanup(self.temp_dir.cleanup)

        objectstore_factory = ObjectStoreFactory()
        state_factory = LinkStateFactory(
            FLWR_IN_MEMORY_DB_NAME, NoOpFederationManager(), objectstore_factory
        )

        self.state = state_factory.state()
        node_id = self.state.create_node("mock_owner", "fake_name", b"pk", 30)
        self.state.acknowledge_node_heartbeat(node_id, 1e3)

        self._server: grpc.Server = run_serverappio_api_grpc(
            SERVERAPPIO_API_DEFAULT_SERVER_ADDRESS,
            state_factory,
            objectstore_factory,
            None,
            superexec_auth_secret=_SUPEREXEC_SECRET,
        )

        # Seed one authenticated task token and reuse it for token-protected RPC
        # checks.
        _, auth_token = self._create_running_run()
        _, self._simulation_token = self._create_running_run(
            run_type=RunType.SIMULATION
        )

        # Create a single base channel and wrap it for authenticated calls.
        self._base_channel = grpc.insecure_channel("localhost:9091")
        self._get_nodes_no_auth = self._base_channel.unary_unary(
            "/flwr.proto.ServerAppIo/GetNodes",
            request_serializer=GetNodesRequest.SerializeToString,
            response_deserializer=GetNodesResponse.FromString,
        )
        auth_channel = grpc.intercept_channel(
            self._base_channel,
            AppIoTokenClientInterceptor(token=auth_token),
            SuperExecAuthClientInterceptor(
                master_secret=_SUPEREXEC_SECRET,
                protected_methods=SERVERAPPIO_SUPEREXEC_METHODS,
            ),
        )
        self._get_nodes = auth_channel.unary_unary(
            "/flwr.proto.ServerAppIo/GetNodes",
            request_serializer=GetNodesRequest.SerializeToString,
            response_deserializer=GetNodesResponse.FromString,
        )

    def tearDown(self) -> None:
        """Stop the gRPC API server."""
        self._base_channel.close()
        self._server.stop(None)

    def _create_running_run(
        self, run_type: str = RunType.SERVER_APP
    ) -> tuple[int, str]:
        run_id = self.state.create_run(
            "", "", "", {}, NOOP_FEDERATION, None, "", run_type
        )
        run = self.state.get_run_info(run_ids=[run_id])[0]
        assert run.primary_task_id is not None
        token = self.state.claim_task(run.primary_task_id)
        assert token is not None
        assert self.state.activate_task(run.primary_task_id)
        return run_id, token

    def test_get_nodes_denied_without_metadata_token(self) -> None:
        """Protected RPC should deny requests missing metadata token."""
        with self.assertRaises(grpc.RpcError) as err:
            self._get_nodes_no_auth.with_call(request=GetNodesRequest())
        assert err.exception.code() == grpc.StatusCode.UNAUTHENTICATED
        assert err.exception.details() == AUTHENTICATION_FAILED_MESSAGE

    def test_get_nodes_denied_with_invalid_metadata_token(self) -> None:
        """Protected RPC should deny requests with invalid metadata token."""
        with self.assertRaises(grpc.RpcError) as err:
            self._get_nodes_no_auth.with_call(
                request=GetNodesRequest(),
                metadata=((TASK_TOKEN_HEADER, "invalid-token"),),
            )
        assert err.exception.code() == grpc.StatusCode.UNAUTHENTICATED
        assert err.exception.details() == AUTHENTICATION_FAILED_MESSAGE

    def test_get_nodes_allows_with_valid_metadata_token(self) -> None:
        """Protected RPC should allow requests with a valid metadata token."""
        response, call = self._get_nodes.with_call(request=GetNodesRequest())

        assert isinstance(response, GetNodesResponse)
        assert call.code() == grpc.StatusCode.OK

    @parameterized.expand(
        [
            (
                "get_nodes",
                "/flwr.proto.ServerAppIo/GetNodes",
                GetNodesRequest(),
                GetNodesResponse.FromString,
            ),
            (
                "push_messages",
                "/flwr.proto.ServerAppIo/PushMessages",
                PushAppMessagesRequest(),
                PushAppMessagesResponse.FromString,
            ),
            (
                "pull_messages",
                "/flwr.proto.ServerAppIo/PullMessages",
                PullAppMessagesRequest(),
                PullAppMessagesResponse.FromString,
            ),
            (
                "push_object",
                "/flwr.proto.ServerAppIo/PushObject",
                PushObjectRequest(),
                PushObjectResponse.FromString,
            ),
            (
                "pull_object",
                "/flwr.proto.ServerAppIo/PullObject",
                PullObjectRequest(),
                PullObjectResponse.FromString,
            ),
            (
                "confirm_message_received",
                "/flwr.proto.ServerAppIo/ConfirmMessageReceived",
                ConfirmMessageReceivedRequest(),
                ConfirmMessageReceivedResponse.FromString,
            ),
        ]
    )  # type: ignore
    def test_serverapp_only_endpoint_denied_for_simulation_run(
        self,
        _case_name: str,
        method: str,
        request: GrpcMessage,
        response_deserializer: Callable[[bytes], object],
    ) -> None:
        """ServerApp-only RPCs should deny simulation-run tokens."""
        rpc = self._base_channel.unary_unary(
            method,
            request_serializer=type(request).SerializeToString,
            response_deserializer=response_deserializer,
        )
        with self.assertRaises(grpc.RpcError) as err:
            rpc.with_call(
                request=request,
                metadata=((TASK_TOKEN_HEADER, self._simulation_token),),
            )
        assert err.exception.code() == grpc.StatusCode.PERMISSION_DENIED

    @parameterized.expand(
        [
            (
                "send_task_heartbeat",
                "/flwr.proto.ServerAppIo/SendTaskHeartbeat",
                SendTaskHeartbeatRequest(),
                SendTaskHeartbeatResponse.FromString,
            ),
            (
                "push_logs",
                "/flwr.proto.ServerAppIo/PushLogs",
                PushLogsRequest(logs=["hello"]),
                PushLogsResponse.FromString,
            ),
        ]
    )  # type: ignore
    def test_shared_task_endpoint_allows_simulation_run(
        self,
        _case_name: str,
        method: str,
        request: GrpcMessage,
        response_deserializer: Callable[[bytes], object],
    ) -> None:
        """Shared task RPCs should still allow simulation-run tokens."""
        rpc = self._base_channel.unary_unary(
            method,
            request_serializer=type(request).SerializeToString,
            response_deserializer=response_deserializer,
        )
        response, call = rpc.with_call(
            request=request,
            metadata=((TASK_TOKEN_HEADER, self._simulation_token),),
        )
        assert response is not None
        assert call.code() == grpc.StatusCode.OK
