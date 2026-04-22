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

from flwr.common.constant import SERVERAPPIO_API_DEFAULT_SERVER_ADDRESS, Status
from flwr.common.typing import RunStatus
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    ListAppsToLaunchRequest,
    ListAppsToLaunchResponse,
    PullAppMessagesRequest,
    PullAppMessagesResponse,
    PushAppMessagesRequest,
    PushAppMessagesResponse,
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
    APP_TOKEN_HEADER,
    AUTHENTICATION_FAILED_MESSAGE,
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

        # Seed one authenticated run/token and reuse it for token-protected RPC checks.
        self._auth_run_id = self._create_running_run()
        auth_token = self.state.create_token(self._auth_run_id)
        assert auth_token is not None
        self._auth_token = auth_token

        self._simulation_run_id = self._create_running_run(run_type=RunType.SIMULATION)
        simulation_token = self.state.create_token(self._simulation_run_id)
        assert simulation_token is not None
        self._simulation_token = simulation_token

        # Create a single base channel and wrap it for authenticated calls.
        self._base_channel = grpc.insecure_channel("localhost:9091")
        self._get_nodes_no_auth = self._base_channel.unary_unary(
            "/flwr.proto.ServerAppIo/GetNodes",
            request_serializer=GetNodesRequest.SerializeToString,
            response_deserializer=GetNodesResponse.FromString,
        )
        self._list_apps_to_launch_no_auth = self._base_channel.unary_unary(
            "/flwr.proto.ServerAppIo/ListAppsToLaunch",
            request_serializer=ListAppsToLaunchRequest.SerializeToString,
            response_deserializer=ListAppsToLaunchResponse.FromString,
        )
        auth_channel = grpc.intercept_channel(
            self._base_channel,
            AppIoTokenClientInterceptor(token=self._auth_token),
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
        self._list_apps_to_launch = auth_channel.unary_unary(
            "/flwr.proto.ServerAppIo/ListAppsToLaunch",
            request_serializer=ListAppsToLaunchRequest.SerializeToString,
            response_deserializer=ListAppsToLaunchResponse.FromString,
        )

    def tearDown(self) -> None:
        """Stop the gRPC API server."""
        self._base_channel.close()
        self._server.stop(None)

    def _create_running_run(self, run_type: str = RunType.SERVER_APP) -> int:
        run_id = self.state.create_run(
            "", "", "", {}, NOOP_FEDERATION, None, "", run_type
        )
        _ = self.state.update_run_status(run_id, RunStatus(Status.STARTING, "", ""))
        _ = self.state.update_run_status(run_id, RunStatus(Status.RUNNING, "", ""))
        return run_id

    def _assert_serverapp_only_endpoint_denied(
        self,
        *,
        method: str,
        request: object,
        response_deserializer: Callable[[bytes], object],
    ) -> None:
        rpc = self._base_channel.unary_unary(
            method,
            request_serializer=request.__class__.SerializeToString,
            response_deserializer=response_deserializer,
        )
        with self.assertRaises(grpc.RpcError) as err:
            rpc.with_call(
                request=request,
                metadata=((APP_TOKEN_HEADER, self._simulation_token),),
            )
        assert err.exception.code() == grpc.StatusCode.PERMISSION_DENIED

    def test_get_nodes_denied_without_metadata_token(self) -> None:
        """Protected RPC should deny requests missing metadata token."""
        with self.assertRaises(grpc.RpcError) as err:
            self._get_nodes_no_auth.with_call(
                request=GetNodesRequest(run_id=self._auth_run_id)
            )
        assert err.exception.code() == grpc.StatusCode.UNAUTHENTICATED
        assert err.exception.details() == AUTHENTICATION_FAILED_MESSAGE

    def test_get_nodes_denied_with_invalid_metadata_token(self) -> None:
        """Protected RPC should deny requests with invalid metadata token."""
        with self.assertRaises(grpc.RpcError) as err:
            self._get_nodes_no_auth.with_call(
                request=GetNodesRequest(run_id=self._auth_run_id),
                metadata=((APP_TOKEN_HEADER, "invalid-token"),),
            )
        assert err.exception.code() == grpc.StatusCode.UNAUTHENTICATED
        assert err.exception.details() == AUTHENTICATION_FAILED_MESSAGE

    def test_get_nodes_allows_with_valid_metadata_token(self) -> None:
        """Protected RPC should allow requests with a valid metadata token."""
        response, call = self._get_nodes.with_call(
            request=GetNodesRequest(run_id=self._auth_run_id)
        )

        assert isinstance(response, GetNodesResponse)
        assert call.code() == grpc.StatusCode.OK

    def test_get_nodes_denied_when_token_targets_different_run(self) -> None:
        """Protected RPC should deny valid tokens used against another run."""
        with self.assertRaises(grpc.RpcError) as err:
            self._get_nodes_no_auth.with_call(
                request=GetNodesRequest(run_id=self._auth_run_id),
                metadata=((APP_TOKEN_HEADER, self._simulation_token),),
            )
        assert err.exception.code() == grpc.StatusCode.PERMISSION_DENIED

    def test_serverapp_only_endpoints_denied_for_simulation_run(self) -> None:
        """ServerApp-only RPCs should deny simulation-run tokens."""
        cases: list[tuple[str, object, Callable[[bytes], object]]] = [
            (
                "/flwr.proto.ServerAppIo/GetNodes",
                GetNodesRequest(run_id=self._simulation_run_id),
                GetNodesResponse.FromString,
            ),
            (
                "/flwr.proto.ServerAppIo/PushMessages",
                PushAppMessagesRequest(run_id=self._simulation_run_id),
                PushAppMessagesResponse.FromString,
            ),
            (
                "/flwr.proto.ServerAppIo/PullMessages",
                PullAppMessagesRequest(run_id=self._simulation_run_id),
                PullAppMessagesResponse.FromString,
            ),
            (
                "/flwr.proto.ServerAppIo/PushObject",
                PushObjectRequest(run_id=self._simulation_run_id),
                PushObjectResponse.FromString,
            ),
            (
                "/flwr.proto.ServerAppIo/PullObject",
                PullObjectRequest(run_id=self._simulation_run_id),
                PullObjectResponse.FromString,
            ),
            (
                "/flwr.proto.ServerAppIo/ConfirmMessageReceived",
                ConfirmMessageReceivedRequest(run_id=self._simulation_run_id),
                ConfirmMessageReceivedResponse.FromString,
            ),
        ]

        for method, request, response_deserializer in cases:
            with self.subTest(method=method):
                self._assert_serverapp_only_endpoint_denied(
                    method=method,
                    request=request,
                    response_deserializer=response_deserializer,
                )

    def test_list_apps_to_launch_denied_without_superexec_metadata(self) -> None:
        """SuperExec RPC should deny requests missing signed metadata."""
        with self.assertRaises(grpc.RpcError) as err:
            self._list_apps_to_launch_no_auth.with_call(
                request=ListAppsToLaunchRequest()
            )
        assert err.exception.code() == grpc.StatusCode.UNAUTHENTICATED
        assert err.exception.details() == AUTHENTICATION_FAILED_MESSAGE

    def test_list_apps_to_launch_allows_with_superexec_metadata(self) -> None:
        """SuperExec RPC should allow requests with valid signed metadata."""
        response, call = self._list_apps_to_launch.with_call(
            request=ListAppsToLaunchRequest()
        )
        assert isinstance(response, ListAppsToLaunchResponse)
        assert call.code() == grpc.StatusCode.OK
