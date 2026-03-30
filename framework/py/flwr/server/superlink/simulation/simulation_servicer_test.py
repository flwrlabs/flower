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
"""SimulationIoServicer tests."""


import tempfile
import unittest
from unittest.mock import Mock

import grpc
from parameterized import parameterized

from flwr.common import Context
from flwr.common.constant import SIMULATIONIO_API_DEFAULT_SERVER_ADDRESS, Status
from flwr.common.serde import context_to_proto, run_status_to_proto
from flwr.common.serde_test import RecordMaker
from flwr.common.typing import RunStatus
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    PushAppOutputsRequest,
    PushAppOutputsResponse,
)
from flwr.proto.federation_config_pb2 import SimulationConfig  # pylint: disable=E0611
from flwr.proto.heartbeat_pb2 import (  # pylint:disable=E0611
    SendAppHeartbeatRequest,
    SendAppHeartbeatResponse,
)
from flwr.proto.run_pb2 import (  # pylint: disable=E0611
    UpdateRunStatusRequest,
    UpdateRunStatusResponse,
)
from flwr.server.superlink.linkstate.linkstate_factory import LinkStateFactory
from flwr.server.superlink.simulation.simulationio_grpc import run_simulationio_api_grpc
from flwr.server.superlink.utils import _STATUS_TO_MSG
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME, NOOP_FEDERATION, RunType
from flwr.supercore.ffs import FfsFactory
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink.federation import NoOpFederationManager


class TestSimulationIoServicer(unittest.TestCase):  # pylint: disable=R0902
    """SimulationIoServicer tests for allowed RunStatuses."""

    def setUp(self) -> None:
        """Initialize mock stub and server interceptor."""
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.addCleanup(self.temp_dir.cleanup)  # Ensures cleanup after test

        state_factory = LinkStateFactory(
            FLWR_IN_MEMORY_DB_NAME, NoOpFederationManager(), ObjectStoreFactory()
        )
        self.state = state_factory.state()
        ffs_factory = FfsFactory(self.temp_dir.name)
        self.ffs = ffs_factory.ffs()

        self.status_to_msg = _STATUS_TO_MSG

        self._server: grpc.Server = run_simulationio_api_grpc(
            SIMULATIONIO_API_DEFAULT_SERVER_ADDRESS,
            state_factory,
            ffs_factory,
            None,
        )

        self._channel = grpc.insecure_channel("localhost:9096")
        self._push_simulation_outputs = self._channel.unary_unary(
            "/flwr.proto.SimulationIo/PushAppOutputs",
            request_serializer=PushAppOutputsRequest.SerializeToString,
            response_deserializer=PushAppOutputsResponse.FromString,
        )
        self._update_run_status = self._channel.unary_unary(
            "/flwr.proto.SimulationIo/UpdateRunStatus",
            request_serializer=UpdateRunStatusRequest.SerializeToString,
            response_deserializer=UpdateRunStatusResponse.FromString,
        )
        self._send_app_heartbeat = self._channel.unary_unary(
            "/flwr.proto.SimulationIo/SendAppHeartbeat",
            request_serializer=SendAppHeartbeatRequest.SerializeToString,
            response_deserializer=SendAppHeartbeatResponse.FromString,
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

    def _create_dummy_run(self) -> int:
        run_id = self.state.create_run(
            "", "", "", {}, NOOP_FEDERATION, SimulationConfig(), "", RunType.SIMULATION
        )
        return run_id

    def test_push_simulation_outputs_successful_if_running(self) -> None:
        """Test `PushAppOutputs` success."""
        # Prepare
        run_id = self._create_dummy_run()
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

        # Transition status to running.
        # PushAppOutputsRequest is only allowed in running status.
        self._transition_run_status(run_id, 2)
        request = PushAppOutputsRequest(
            token=token, run_id=run_id, context=context_to_proto(context)
        )

        # Execute
        response, call = self._push_simulation_outputs.with_call(request=request)

        # Assert
        assert isinstance(response, PushAppOutputsResponse)
        assert grpc.StatusCode.OK == call.code()

    def _assert_push_simulation_outputs_not_allowed(
        self, token: str, context: Context
    ) -> None:
        """Assert `PushAppOutputs` not allowed."""
        run_id = self.state.get_run_id_by_token(token)
        assert run_id is not None, "Invalid token is provided."
        run_status = self.state.get_run_status({run_id})[run_id]
        request = PushAppOutputsRequest(
            token=token, run_id=run_id, context=context_to_proto(context)
        )

        with self.assertRaises(grpc.RpcError) as e:
            self._push_simulation_outputs.with_call(request=request)
        assert e.exception.code() == grpc.StatusCode.PERMISSION_DENIED
        assert e.exception.details() == self.status_to_msg[run_status.status]

    @parameterized.expand(
        [
            (0,),  # Test not successful if RunStatus is pending.
            (1,),  # Test not successful if RunStatus is starting.
            (3,),  # Test not successful if RunStatus is finished.
        ]
    )  # type: ignore
    def test_push_simulation_outputs_not_successful_if_not_running(
        self, num_transitions: int
    ) -> None:
        """Test `PushAppOutputs` not successful if RunStatus is not running."""
        # Prepare
        run_id = self._create_dummy_run()
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
        self._assert_push_simulation_outputs_not_allowed(token, context)

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
        run_id = self._create_dummy_run()
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
        run_id = self._create_dummy_run()
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
