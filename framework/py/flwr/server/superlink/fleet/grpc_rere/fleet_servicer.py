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
"""Fleet API gRPC request-response servicer."""


import grpc

from flwr.proto import fleet_pb2_grpc  # pylint: disable=E0611
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
from flwr.server.superlink.linkstate import LinkStateFactory
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink.servicer.fleet import fleet_handlers


class FleetServicer(fleet_pb2_grpc.FleetServicer):
    """Fleet API servicer."""

    def __init__(
        self,
        state_factory: LinkStateFactory,
        objectstore_factory: ObjectStoreFactory,
        enable_supernode_auth: bool,
    ) -> None:
        self.state_factory = state_factory
        self.objectstore_factory = objectstore_factory
        self.enable_supernode_auth = enable_supernode_auth

    def RegisterNode(
        self, request: RegisterNodeFleetRequest, context: grpc.ServicerContext
    ) -> RegisterNodeFleetResponse:
        """Register a node."""
        return fleet_handlers.register_node(
            request=request,
            state=self.state_factory.state(),
            enable_supernode_auth=self.enable_supernode_auth,
        )

    def ActivateNode(
        self, request: ActivateNodeRequest, context: grpc.ServicerContext
    ) -> ActivateNodeResponse:
        """Activate a node."""
        return fleet_handlers.activate_node(
            request=request,
            state=self.state_factory.state(),
        )

    def DeactivateNode(
        self, request: DeactivateNodeRequest, context: grpc.ServicerContext
    ) -> DeactivateNodeResponse:
        """Deactivate a node."""
        return fleet_handlers.deactivate_node(
            request=request,
            state=self.state_factory.state(),
        )

    def UnregisterNode(
        self, request: UnregisterNodeFleetRequest, context: grpc.ServicerContext
    ) -> UnregisterNodeFleetResponse:
        """Unregister a node."""
        return fleet_handlers.unregister_node(
            request=request,
            state=self.state_factory.state(),
            enable_supernode_auth=self.enable_supernode_auth,
        )

    def SendNodeHeartbeat(
        self, request: SendNodeHeartbeatRequest, context: grpc.ServicerContext
    ) -> SendNodeHeartbeatResponse:
        """."""
        return fleet_handlers.send_node_heartbeat(
            request=request,
            state=self.state_factory.state(),
        )

    def PullMessages(
        self, request: PullMessagesRequest, context: grpc.ServicerContext
    ) -> PullMessagesResponse:
        """Pull Messages."""
        return fleet_handlers.pull_messages(
            request=request,
            state=self.state_factory.state(),
            store=self.objectstore_factory.store(),
        )

    def PushMessages(
        self, request: PushMessagesRequest, context: grpc.ServicerContext
    ) -> PushMessagesResponse:
        """Push Messages."""
        return fleet_handlers.push_messages(
            request=request,
            state=self.state_factory.state(),
        )

    def GetRun(
        self, request: GetRunRequest, context: grpc.ServicerContext
    ) -> GetRunResponse:
        """Get run information."""
        return fleet_handlers.get_run(
            request=request,
            state=self.state_factory.state(),
        )

    def GetFab(
        self, request: GetFabRequest, context: grpc.ServicerContext
    ) -> GetFabResponse:
        """Get FAB."""
        return fleet_handlers.get_fab(
            request=request,
            state=self.state_factory.state(),
        )

    def PushObject(
        self, request: PushObjectRequest, context: grpc.ServicerContext
    ) -> PushObjectResponse:
        """Push an object to the ObjectStore."""
        return fleet_handlers.push_object(
            request=request,
            state=self.state_factory.state(),
        )

    def PullObject(
        self, request: PullObjectRequest, context: grpc.ServicerContext
    ) -> PullObjectResponse:
        """Pull an object from the ObjectStore."""
        return fleet_handlers.pull_object(
            request=request,
            state=self.state_factory.state(),
        )

    def ConfirmMessageReceived(
        self, request: ConfirmMessageReceivedRequest, context: grpc.ServicerContext
    ) -> ConfirmMessageReceivedResponse:
        """Confirm message received."""
        return fleet_handlers.confirm_message_received(
            request=request,
            state=self.state_factory.state(),
            store=self.objectstore_factory.store(),
        )
