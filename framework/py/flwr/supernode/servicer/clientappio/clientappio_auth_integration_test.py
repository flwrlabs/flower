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
"""ClientAppIo auth interceptor integration tests."""


import tempfile
import unittest

import grpc

from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    ListAppsToLaunchRequest,
    ListAppsToLaunchResponse,
)
from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    PullObjectRequest,
    PullObjectResponse,
)
from flwr.supercore.interceptors import (
    APP_TOKEN_HEADER,
    AUTHENTICATION_FAILED_MESSAGE,
    SuperExecAuthClientInterceptor,
)
from flwr.supercore.interceptors.superexec_auth_interceptor import (
    CLIENTAPPIO_SUPEREXEC_METHODS,
)
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supernode.nodestate import NodeStateFactory
from flwr.supernode.start_client_internal import run_clientappio_api_grpc

_SUPEREXEC_SECRET = b"test-superexec-secret"


def _to_loopback_target(bound_address: str) -> str:
    """Convert a bound server address to a client-connectable loopback target."""
    _host, port = bound_address.rsplit(":", maxsplit=1)
    return f"127.0.0.1:{port}"


class TestClientAppIoAuthIntegration(unittest.TestCase):
    """Integration tests for ClientAppIo token-auth interceptor behavior."""

    def setUp(self) -> None:
        """Start the ClientAppIo gRPC API without client-side auth helpers."""
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.addCleanup(self.temp_dir.cleanup)

        objectstore_factory = ObjectStoreFactory()
        state_factory = NodeStateFactory(objectstore_factory=objectstore_factory)

        state = state_factory.state()
        token = state.create_token(99)
        assert token is not None
        self.valid_token = token

        self._server: grpc.Server = run_clientappio_api_grpc(
            address="127.0.0.1:0",
            state_factory=state_factory,
            objectstore_factory=objectstore_factory,
            certificates=None,
            superexec_auth_secret=_SUPEREXEC_SECRET,
        )

        server_target = _to_loopback_target(self._server.bound_address)
        channel = grpc.insecure_channel(server_target)
        self._pull_object = channel.unary_unary(
            "/flwr.proto.ClientAppIo/PullObject",
            request_serializer=PullObjectRequest.SerializeToString,
            response_deserializer=PullObjectResponse.FromString,
        )
        self._list_apps_to_launch = channel.unary_unary(
            "/flwr.proto.ClientAppIo/ListAppsToLaunch",
            request_serializer=ListAppsToLaunchRequest.SerializeToString,
            response_deserializer=ListAppsToLaunchResponse.FromString,
        )
        superexec_channel = grpc.intercept_channel(
            grpc.insecure_channel(server_target),
            SuperExecAuthClientInterceptor(
                master_secret=_SUPEREXEC_SECRET,
                protected_methods=CLIENTAPPIO_SUPEREXEC_METHODS,
            ),
        )
        self._list_apps_to_launch_superexec = superexec_channel.unary_unary(
            "/flwr.proto.ClientAppIo/ListAppsToLaunch",
            request_serializer=ListAppsToLaunchRequest.SerializeToString,
            response_deserializer=ListAppsToLaunchResponse.FromString,
        )

    def tearDown(self) -> None:
        """Stop the gRPC API server."""
        self._server.stop(None)

    def test_pull_object_denied_without_metadata_token(self) -> None:
        """Protected RPC should deny requests missing metadata token."""
        with self.assertRaises(grpc.RpcError) as err:
            self._pull_object.with_call(request=PullObjectRequest(object_id="obj-1"))
        assert err.exception.code() == grpc.StatusCode.UNAUTHENTICATED
        assert err.exception.details() == AUTHENTICATION_FAILED_MESSAGE

    def test_pull_object_denied_with_invalid_metadata_token(self) -> None:
        """Protected RPC should deny requests with invalid metadata token."""
        with self.assertRaises(grpc.RpcError) as err:
            self._pull_object.with_call(
                request=PullObjectRequest(object_id="obj-2"),
                metadata=((APP_TOKEN_HEADER, "invalid-token"),),
            )
        assert err.exception.code() == grpc.StatusCode.UNAUTHENTICATED
        assert err.exception.details() == AUTHENTICATION_FAILED_MESSAGE

    def test_pull_object_allows_with_valid_metadata_token(self) -> None:
        """Protected RPC should allow requests with valid metadata token."""
        response, call = self._pull_object.with_call(
            request=PullObjectRequest(object_id="obj-3"),
            metadata=((APP_TOKEN_HEADER, self.valid_token),),
        )

        assert isinstance(response, PullObjectResponse)
        assert call.code() == grpc.StatusCode.OK

    def test_list_apps_to_launch_denied_without_superexec_metadata(self) -> None:
        """SuperExec RPC should deny requests missing signed metadata."""
        with self.assertRaises(grpc.RpcError) as err:
            self._list_apps_to_launch.with_call(request=ListAppsToLaunchRequest())
        assert err.exception.code() == grpc.StatusCode.UNAUTHENTICATED
        assert err.exception.details() == AUTHENTICATION_FAILED_MESSAGE

    def test_list_apps_to_launch_allows_with_superexec_metadata(self) -> None:
        """SuperExec RPC should allow requests with valid signed metadata."""
        response, call = self._list_apps_to_launch_superexec.with_call(
            request=ListAppsToLaunchRequest()
        )
        assert isinstance(response, ListAppsToLaunchResponse)
        assert call.code() == grpc.StatusCode.OK


class TestClientAppIoAuthIntegrationWithoutSuperExecSecret(unittest.TestCase):
    """Integration tests for ClientAppIo when SuperExec auth is disabled."""

    def setUp(self) -> None:
        """Start the ClientAppIo API with only token interceptor enabled."""
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.addCleanup(self.temp_dir.cleanup)

        objectstore_factory = ObjectStoreFactory()
        state_factory = NodeStateFactory(objectstore_factory=objectstore_factory)

        self._server: grpc.Server = run_clientappio_api_grpc(
            address="127.0.0.1:0",
            state_factory=state_factory,
            objectstore_factory=objectstore_factory,
            certificates=None,
            superexec_auth_secret=None,
        )

        channel = grpc.insecure_channel(_to_loopback_target(self._server.bound_address))
        self._list_apps_to_launch = channel.unary_unary(
            "/flwr.proto.ClientAppIo/ListAppsToLaunch",
            request_serializer=ListAppsToLaunchRequest.SerializeToString,
            response_deserializer=ListAppsToLaunchResponse.FromString,
        )

    def tearDown(self) -> None:
        """Stop the gRPC API server."""
        self._server.stop(None)

    def test_list_apps_to_launch_allows_without_superexec_metadata(self) -> None:
        """No SuperExec signing should be required when auth is disabled."""
        response, call = self._list_apps_to_launch.with_call(
            request=ListAppsToLaunchRequest()
        )
        assert isinstance(response, ListAppsToLaunchResponse)
        assert call.code() == grpc.StatusCode.OK
