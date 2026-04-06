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
"""ServerAppIo token lifecycle integration tests."""


import tempfile
import unittest

import grpc

from flwr.common import Context
from flwr.common.constant import SERVERAPPIO_API_DEFAULT_SERVER_ADDRESS, Status
from flwr.common.serde import context_to_proto, run_status_to_proto
from flwr.common.serde_test import RecordMaker
from flwr.common.typing import RunStatus
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    PushAppOutputsRequest,
    PushAppOutputsResponse,
)
from flwr.proto.run_pb2 import (  # pylint: disable=E0611
    UpdateRunStatusRequest,
    UpdateRunStatusResponse,
)
from flwr.server.superlink.linkstate.linkstate_factory import LinkStateFactory
from flwr.server.superlink.serverappio.serverappio_grpc import run_serverappio_api_grpc
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME, NOOP_FEDERATION, RunType
from flwr.supercore.ffs import FfsFactory
from flwr.supercore.interceptors import APP_TOKEN_HEADER
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink.federation import NoOpFederationManager


class TestServerAppIoTokenLifecycleIntegration(unittest.TestCase):
    """Integration tests for token deletion timing on ServerAppIo."""

    def setUp(self) -> None:
        """Start the ServerAppIo gRPC API."""
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.addCleanup(self.temp_dir.cleanup)

        objectstore_factory = ObjectStoreFactory()
        state_factory = LinkStateFactory(
            FLWR_IN_MEMORY_DB_NAME, NoOpFederationManager(), objectstore_factory
        )
        ffs_factory = FfsFactory(self.temp_dir.name)

        self.state = state_factory.state()
        node_id = self.state.create_node("mock_owner", "fake_name", b"pk", 30)
        self.state.acknowledge_node_heartbeat(node_id, 1e3)

        self._server: grpc.Server = run_serverappio_api_grpc(
            SERVERAPPIO_API_DEFAULT_SERVER_ADDRESS,
            state_factory,
            ffs_factory,
            objectstore_factory,
            None,
        )

        channel = grpc.insecure_channel("localhost:9091")
        self._push_app_outputs = channel.unary_unary(
            "/flwr.proto.ServerAppIo/PushAppOutputs",
            request_serializer=PushAppOutputsRequest.SerializeToString,
            response_deserializer=PushAppOutputsResponse.FromString,
        )
        self._update_run_status = channel.unary_unary(
            "/flwr.proto.ServerAppIo/UpdateRunStatus",
            request_serializer=UpdateRunStatusRequest.SerializeToString,
            response_deserializer=UpdateRunStatusResponse.FromString,
        )

    def tearDown(self) -> None:
        """Stop the gRPC API server."""
        self._server.stop(None)

    def _create_running_run_and_token(self) -> tuple[int, str]:
        run_id = self.state.create_run(
            "", "", "", {}, NOOP_FEDERATION, None, "", RunType.SERVER_APP
        )
        _ = self.state.update_run_status(run_id, RunStatus(Status.STARTING, "", ""))
        _ = self.state.update_run_status(run_id, RunStatus(Status.RUNNING, "", ""))
        token = self.state.create_token(run_id)
        assert token is not None
        return run_id, token

    def test_push_app_outputs_keeps_token(self) -> None:
        """`PushAppOutputs` should not delete the token."""
        run_id, token = self._create_running_run_and_token()
        maker = RecordMaker()
        context = Context(
            run_id=run_id,
            node_id=0,
            node_config=maker.user_config(),
            state=maker.recorddict(1, 1, 1),
            run_config=maker.user_config(),
        )
        request = PushAppOutputsRequest(
            token=token,
            run_id=run_id,
            context=context_to_proto(context),
        )

        response, call = self._push_app_outputs.with_call(
            request=request,
            metadata=((APP_TOKEN_HEADER, token),),
        )

        assert isinstance(response, PushAppOutputsResponse)
        assert call.code() == grpc.StatusCode.OK
        assert self.state.get_run_id_by_token(token) == run_id

    def test_update_run_status_finished_deletes_token(self) -> None:
        """`UpdateRunStatus(FINISHED)` should delete the token."""
        run_id, token = self._create_running_run_and_token()
        request = UpdateRunStatusRequest(
            run_id=run_id,
            run_status=run_status_to_proto(RunStatus(Status.FINISHED, "", "")),
        )

        response, call = self._update_run_status.with_call(
            request=request,
            metadata=((APP_TOKEN_HEADER, token),),
        )

        assert isinstance(response, UpdateRunStatusResponse)
        assert call.code() == grpc.StatusCode.OK
        assert self.state.get_run_id_by_token(token) is None
