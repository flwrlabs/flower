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
"""HTTP client for the Fleet API."""

from flwr.common.constant import HEARTBEAT_CALL_TIMEOUT
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
    ConfirmMessageReceivedRequest,
    ConfirmMessageReceivedResponse,
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
from flwr.supercore.protobuf.client import ProtobufClient


# Match the method names defined by the Fleet protobuf service.
# pylint: disable=invalid-name
class FleetHttpClient(ProtobufClient):  # pylint: disable=too-many-public-methods
    """Protobuf-over-HTTP client for the Fleet API."""

    def RegisterNode(
        self, request: RegisterNodeFleetRequest
    ) -> RegisterNodeFleetResponse:
        """Register a SuperNode."""
        return self._unary_unary(
            path="/v1/fleet/register-node",
            rpc_method="/flwr.proto.Fleet/RegisterNode",
            request=request,
            response_type=RegisterNodeFleetResponse,
        )

    def ActivateNode(self, request: ActivateNodeRequest) -> ActivateNodeResponse:
        """Activate a SuperNode."""
        return self._unary_unary(
            path="/v1/fleet/activate-node",
            rpc_method="/flwr.proto.Fleet/ActivateNode",
            request=request,
            response_type=ActivateNodeResponse,
        )

    def DeactivateNode(self, request: DeactivateNodeRequest) -> DeactivateNodeResponse:
        """Deactivate a SuperNode."""
        return self._unary_unary(
            path="/v1/fleet/deactivate-node",
            rpc_method="/flwr.proto.Fleet/DeactivateNode",
            request=request,
            response_type=DeactivateNodeResponse,
        )

    def UnregisterNode(
        self, request: UnregisterNodeFleetRequest
    ) -> UnregisterNodeFleetResponse:
        """Unregister a SuperNode."""
        return self._unary_unary(
            path="/v1/fleet/unregister-node",
            rpc_method="/flwr.proto.Fleet/UnregisterNode",
            request=request,
            response_type=UnregisterNodeFleetResponse,
        )

    def SendNodeHeartbeat(
        self, request: SendNodeHeartbeatRequest
    ) -> SendNodeHeartbeatResponse:
        """Send a SuperNode heartbeat."""
        return self._unary_unary(
            path="/v1/fleet/send-node-heartbeat",
            rpc_method="/flwr.proto.Fleet/SendNodeHeartbeat",
            request=request,
            response_type=SendNodeHeartbeatResponse,
            # Before enabling HTTP/2, verify this timeout with other active streams.
            timeout=HEARTBEAT_CALL_TIMEOUT,
        )

    def PullMessages(self, request: PullMessagesRequest) -> PullMessagesResponse:
        """Pull messages for a SuperNode."""
        return self._unary_unary(
            path="/v1/fleet/pull-messages",
            rpc_method="/flwr.proto.Fleet/PullMessages",
            request=request,
            response_type=PullMessagesResponse,
        )

    def PushMessages(self, request: PushMessagesRequest) -> PushMessagesResponse:
        """Push messages from a SuperNode."""
        return self._unary_unary(
            path="/v1/fleet/push-messages",
            rpc_method="/flwr.proto.Fleet/PushMessages",
            request=request,
            response_type=PushMessagesResponse,
        )

    def GetRun(self, request: GetRunRequest) -> GetRunResponse:
        """Get a run."""
        return self._unary_unary(
            path="/v1/fleet/get-run",
            rpc_method="/flwr.proto.Fleet/GetRun",
            request=request,
            response_type=GetRunResponse,
        )

    def GetFab(self, request: GetFabRequest) -> GetFabResponse:
        """Get a FAB."""
        return self._unary_unary(
            path="/v1/fleet/get-fab",
            rpc_method="/flwr.proto.Fleet/GetFab",
            request=request,
            response_type=GetFabResponse,
        )

    def PushObject(self, request: PushObjectRequest) -> PushObjectResponse:
        """Push an object to the SuperLink."""
        return self._unary_unary(
            path="/v1/fleet/push-object",
            rpc_method="/flwr.proto.Fleet/PushObject",
            request=request,
            response_type=PushObjectResponse,
        )

    def PullObject(self, request: PullObjectRequest) -> PullObjectResponse:
        """Pull an object from the SuperLink."""
        return self._unary_unary(
            path="/v1/fleet/pull-object",
            rpc_method="/flwr.proto.Fleet/PullObject",
            request=request,
            response_type=PullObjectResponse,
        )

    def ConfirmMessageReceived(
        self, request: ConfirmMessageReceivedRequest
    ) -> ConfirmMessageReceivedResponse:
        """Confirm receipt of an object."""
        return self._unary_unary(
            path="/v1/fleet/confirm-message-received",
            rpc_method="/flwr.proto.Fleet/ConfirmMessageReceived",
            request=request,
            response_type=ConfirmMessageReceivedResponse,
        )
