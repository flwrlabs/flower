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
"""SuperLink Runtime API auth interceptor integration tests."""


import tempfile
import unittest
from collections.abc import Callable

import grpc
from google.protobuf.message import Message as GrpcMessage
from parameterized import parameterized

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
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    GetConnectorRequest,
    GetConnectorResponse,
    GetNodesRequest,
    GetNodesResponse,
    PullAppMessagesRequest,
    PullAppMessagesResponse,
    PushAppMessagesRequest,
    PushAppMessagesResponse,
    SendTaskHeartbeatRequest,
    SendTaskHeartbeatResponse,
)
from flwr.server.superlink.linkstate.linkstate_factory import LinkStateFactory
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME, NOOP_FEDERATION_ID, TaskType
from flwr.supercore.interceptors import (
    AUTHENTICATION_FAILED_MESSAGE,
    TASK_TOKEN_HEADER,
    RuntimeTokenClientInterceptor,
    SuperExecAuthClientInterceptor,
)
from flwr.supercore.interceptors.superexec_auth_interceptor import (
    RUNTIME_SUPEREXEC_METHODS,
)
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink.federation import NoOpFederationManager
from flwr.superlink.servicer.runtime.runtime_grpc import run_runtime_api_grpc

_SUPEREXEC_SECRET = b"test-superexec-secret"
_LEGACY_RUNTIME_API_ADDRESS = "0.0.0.0:9091"


class TestSuperLinkRuntimeAuthIntegration(unittest.TestCase):  # pylint: disable=R0902
    """Integration tests for SuperLink Runtime token-auth behavior."""

    def setUp(self) -> None:
        """Start the Runtime API without client-side auth helpers."""
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.addCleanup(self.temp_dir.cleanup)

        objectstore_factory = ObjectStoreFactory()
        state_factory = LinkStateFactory(
            FLWR_IN_MEMORY_DB_NAME, NoOpFederationManager(), objectstore_factory
        )

        self.state = state_factory.state()
        node_id = self.state.create_node("mock_owner", "fake_name", b"pk", 30)
        self.state.acknowledge_node_heartbeat(node_id, 1e3)

        self._server: grpc.Server = run_runtime_api_grpc(
            _LEGACY_RUNTIME_API_ADDRESS,
            state_factory,
            objectstore_factory,
            None,
            superexec_auth_secret=_SUPEREXEC_SECRET,
        )

        # Seed one authenticated task token and reuse it for token-protected RPC
        # checks.
        _, auth_token = self._create_running_run()
        _, self._simulation_token = self._create_running_run(
            primary_task_type=TaskType.SIMULATION
        )

        # Create a single base channel and wrap it for authenticated calls.
        self._base_channel = grpc.insecure_channel("localhost:9091")
        self._get_nodes_no_auth = self._base_channel.unary_unary(
            "/flwr.proto.Runtime/GetNodes",
            request_serializer=GetNodesRequest.SerializeToString,
            response_deserializer=GetNodesResponse.FromString,
        )
        self._get_connector_no_auth = self._base_channel.unary_unary(
            "/flwr.proto.Runtime/GetConnector",
            request_serializer=GetConnectorRequest.SerializeToString,
            response_deserializer=GetConnectorResponse.FromString,
        )
        auth_channel = grpc.intercept_channel(
            self._base_channel,
            RuntimeTokenClientInterceptor(token=auth_token),
            SuperExecAuthClientInterceptor(
                master_secret=_SUPEREXEC_SECRET,
                protected_methods=RUNTIME_SUPEREXEC_METHODS,
            ),
        )
        self._get_nodes = auth_channel.unary_unary(
            "/flwr.proto.Runtime/GetNodes",
            request_serializer=GetNodesRequest.SerializeToString,
            response_deserializer=GetNodesResponse.FromString,
        )

    def tearDown(self) -> None:
        """Stop the gRPC API server."""
        self._base_channel.close()
        self._server.stop(None)

    def _create_running_run(
        self, primary_task_type: str = TaskType.SERVER_APP
    ) -> tuple[int, str]:
        run_id = self.state.create_run(
            "", "", "", {}, NOOP_FEDERATION_ID, None, "", primary_task_type
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

    def test_get_connector_requires_and_uses_connector_task_token(self) -> None:
        """GetConnector should derive credential access from its task token."""
        with self.assertRaises(grpc.RpcError) as unauthenticated:
            self._get_connector_no_auth.with_call(request=GetConnectorRequest())
        assert unauthenticated.exception.code() == grpc.StatusCode.UNAUTHENTICATED

        run_id = self.state.create_run(
            "",
            "",
            "",
            {},
            NOOP_FEDERATION_ID,
            None,
            "account-a",
            TaskType.AGENT_APP,
            connector_refs=["notion"],
        )
        task_id = self.state.create_task(
            TaskType.CONNECTOR,
            run_id,
            connector_ref="notion",
        )
        assert task_id is not None
        token = self.state.claim_task(task_id)
        assert token is not None
        assert self.state.activate_task(task_id)
        assert self.state.upsert_connector(
            flwr_aid="account-a",
            connector_ref="notion",
            credentials_json='{"token":"secret"}',
            config_json="{}",
        )

        response, call = self._get_connector_no_auth.with_call(
            request=GetConnectorRequest(),
            metadata=((TASK_TOKEN_HEADER, token),),
        )

        assert call.code() == grpc.StatusCode.OK
        assert response == GetConnectorResponse(
            connector_ref="notion",
            credentials_json='{"token":"secret"}',
            config_json="{}",
        )

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
                "/flwr.proto.Runtime/GetNodes",
                GetNodesRequest(),
                GetNodesResponse.FromString,
            ),
            (
                "push_messages",
                "/flwr.proto.Runtime/PushMessages",
                PushAppMessagesRequest(),
                PushAppMessagesResponse.FromString,
            ),
            (
                "pull_messages",
                "/flwr.proto.Runtime/PullMessages",
                PullAppMessagesRequest(),
                PullAppMessagesResponse.FromString,
            ),
            (
                "push_object",
                "/flwr.proto.Runtime/PushObject",
                PushObjectRequest(),
                PushObjectResponse.FromString,
            ),
            (
                "pull_object",
                "/flwr.proto.Runtime/PullObject",
                PullObjectRequest(),
                PullObjectResponse.FromString,
            ),
            (
                "confirm_message_received",
                "/flwr.proto.Runtime/ConfirmMessageReceived",
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
                "/flwr.proto.Runtime/SendTaskHeartbeat",
                SendTaskHeartbeatRequest(),
                SendTaskHeartbeatResponse.FromString,
            ),
            (
                "push_logs",
                "/flwr.proto.Runtime/PushLogs",
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
