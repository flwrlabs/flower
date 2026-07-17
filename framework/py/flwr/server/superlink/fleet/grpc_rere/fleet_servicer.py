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


import threading
from logging import DEBUG, INFO

import grpc
from google.protobuf.json_format import MessageToDict

from flwr.common.logger import log
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
from flwr.server.superlink.fleet.message_handler import message_handler
from flwr.server.superlink.linkstate import LinkStateFactory
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.run import InvalidRunStatusException


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
        self.lock = threading.Lock()

    def RegisterNode(
        self, request: RegisterNodeFleetRequest, context: grpc.ServicerContext
    ) -> RegisterNodeFleetResponse:
        """Register a node."""
        error_context = (
            f"Attempted to register SuperNode with public key: {request.public_key!r}"
        )

        # Prevent registration when SuperNode authentication is enabled
        if self.enable_supernode_auth:
            raise FlowerError(
                ApiErrorCode.FLEET_SUPERNODE_REGISTRATION_DISABLED,
                error_context,
            )

        try:
            response = message_handler.register_node(
                request=request,
                state=self.state_factory.state(),
            )
            log(DEBUG, "[Fleet.RegisterNode] Registered node_id=%s", response.node_id)
            return response
        except ValueError as e:
            # Public key already in use
            # This should NEVER happen due to the public keys should be automatically
            # generated and unique for each SuperNode instance.
            raise FlowerError(
                ApiErrorCode.PUBLIC_KEY_ALREADY_IN_USE,
                error_context,
            ) from e

    def ActivateNode(
        self, request: ActivateNodeRequest, context: grpc.ServicerContext
    ) -> ActivateNodeResponse:
        """Activate a node."""
        error_context = (
            f"Attempted to register SuperNode with public key: {request.public_key!r}"
        )

        try:
            response = message_handler.activate_node(
                request=request,
                state=self.state_factory.state(),
            )
            log(INFO, "[Fleet.ActivateNode] Activated node_id=%s", response.node_id)
            return response
        except message_handler.InvalidHeartbeatIntervalError as e:
            # Heartbeat interval is invalid
            raise FlowerError(
                ApiErrorCode.FLEET_INVALID_HEARTBEAT_INTERVAL,
                f"{error_context}, exception: {str(e)}",
            ) from e
        except ValueError as e:
            raise FlowerError(
                ApiErrorCode.FLEET_NODE_ACTIVATION_FAILED,
                f"{error_context}, exception: {str(e)}",
            ) from e

    def DeactivateNode(
        self, request: DeactivateNodeRequest, context: grpc.ServicerContext
    ) -> DeactivateNodeResponse:
        """Deactivate a node."""
        try:
            response = message_handler.deactivate_node(
                request=request,
                state=self.state_factory.state(),
            )
            log(INFO, "[Fleet.DeactivateNode] Deactivated node_id=%s", request.node_id)
            return response
        except ValueError as e:
            raise FlowerError(
                ApiErrorCode.FLEET_NODE_DEACTIVATION_FAILED,
                f"SuperNode {request.node_id}, exception: {str(e)}",
            ) from e

    def UnregisterNode(
        self, request: UnregisterNodeFleetRequest, context: grpc.ServicerContext
    ) -> UnregisterNodeFleetResponse:
        """Unregister a node."""
        error_context = f"node_id={request.node_id}"

        # Prevent unregistration when SuperNode authentication is enabled
        if self.enable_supernode_auth:
            raise FlowerError(
                ApiErrorCode.FLEET_SUPERNODE_UNREGISTRATION_DISABLED,
                f"{error_context}, SuperNode unregistration is disabled through "
                "Fleet API.",
            )

        try:
            response = message_handler.unregister_node(
                request=request,
                state=self.state_factory.state(),
            )
            log(
                DEBUG, "[Fleet.UnregisterNode] Unregistered node_id=%s", request.node_id
            )
            return response
        except ValueError as e:
            raise FlowerError(
                ApiErrorCode.FLEET_NODE_UNREGISTRATION_FAILED,
                f"{error_context}, exception: {str(e)}",
            ) from e

    def SendNodeHeartbeat(
        self, request: SendNodeHeartbeatRequest, context: grpc.ServicerContext
    ) -> SendNodeHeartbeatResponse:
        """."""
        log(DEBUG, "[Fleet.SendNodeHeartbeat] Request: %s", MessageToDict(request))
        try:
            return message_handler.send_node_heartbeat(
                request=request,
                state=self.state_factory.state(),
            )
        except message_handler.InvalidHeartbeatIntervalError as e:
            # Heartbeat interval is invalid
            raise FlowerError(
                ApiErrorCode.FLEET_INVALID_HEARTBEAT_INTERVAL,
                str(e),
            ) from e

    def PullMessages(
        self, request: PullMessagesRequest, context: grpc.ServicerContext
    ) -> PullMessagesResponse:
        """Pull Messages."""
        log(INFO, "[Fleet.PullMessages] node_id=%s", request.node.node_id)
        log(DEBUG, "[Fleet.PullMessages] Request: %s", MessageToDict(request))
        return message_handler.pull_messages(
            request=request,
            state=self.state_factory.state(),
            store=self.objectstore_factory.store(),
        )

    def PushMessages(
        self, request: PushMessagesRequest, context: grpc.ServicerContext
    ) -> PushMessagesResponse:
        """Push Messages."""
        if request.messages_list:
            log(
                INFO,
                "[Fleet.PushMessages] Push replies from node_id=%s",
                request.messages_list[0].metadata.src_node_id,
            )
        else:
            log(INFO, "[Fleet.PushMessages] No replies to push")

        try:
            res = message_handler.push_messages(
                request=request,
                state=self.state_factory.state(),
            )
        except InvalidRunStatusException as e:
            raise FlowerError(
                ApiErrorCode.FLEET_RUN_STATUS_NOT_ALLOWED,
                f"SuperNode {request.node.node_id}, exception: {e.message}",
            ) from e

        return res

    def GetRun(
        self, request: GetRunRequest, context: grpc.ServicerContext
    ) -> GetRunResponse:
        """Get run information."""
        log(INFO, "[Fleet.GetRun] Requesting `Run` for run_id=%s", request.run_id)
        error_context = f"SuperNode {request.node.node_id}, run_id={request.run_id}"

        try:
            res = message_handler.get_run(
                request=request,
                state=self.state_factory.state(),
            )
        except InvalidRunStatusException as e:
            raise FlowerError(
                ApiErrorCode.FLEET_RUN_STATUS_NOT_ALLOWED,
                f"{error_context}, exception: {e.message}",
            ) from e
        except ValueError as e:
            raise FlowerError(
                ApiErrorCode.FLEET_GET_RUN_FAILED,
                f"{error_context}, exception: {str(e)}",
            ) from e

        return res

    def GetFab(
        self, request: GetFabRequest, context: grpc.ServicerContext
    ) -> GetFabResponse:
        """Get FAB."""
        log(INFO, "[Fleet.GetFab] Requesting FAB for fab_hash=%s", request.hash_str)
        error_context = (
            f"SuperNode {request.node.node_id}, run_id={request.run_id}, "
            f"fab_hash={request.hash_str}"
        )
        try:
            res = message_handler.get_fab(
                request=request,
                state=self.state_factory.state(),
            )
        except InvalidRunStatusException as e:
            raise FlowerError(
                ApiErrorCode.FLEET_RUN_STATUS_NOT_ALLOWED,
                f"{error_context}, exception: {e.message}",
            ) from e
        except ValueError as e:
            raise FlowerError(
                ApiErrorCode.FLEET_GET_FAB_FAILED,
                f"{error_context}, exception: {str(e)}",
            ) from e

        return res

    def PushObject(
        self, request: PushObjectRequest, context: grpc.ServicerContext
    ) -> PushObjectResponse:
        """Push an object to the ObjectStore."""
        log(
            DEBUG,
            "[Fleet.PushObject] Push Object with object_id=%s",
            request.object_id,
        )
        error_context = (
            f"SuperNode {request.node.node_id}, run_id={request.run_id}, "
            f"object_id={request.object_id}"
        )

        try:
            # Insert in Store
            res = message_handler.push_object(
                request=request,
                state=self.state_factory.state(),
            )
        except InvalidRunStatusException as e:
            raise FlowerError(
                ApiErrorCode.FLEET_RUN_STATUS_NOT_ALLOWED,
                f"{error_context}, exception: {e.message}",
                public_details=f"Object_id: {request.object_id}",
            ) from e
        return res

    def PullObject(
        self, request: PullObjectRequest, context: grpc.ServicerContext
    ) -> PullObjectResponse:
        """Pull an object from the ObjectStore."""
        log(
            DEBUG,
            "[Fleet.PullObject] Pull Object with object_id=%s",
            request.object_id,
        )
        error_context = (
            f"SuperNode {request.node.node_id}, run_id={request.run_id}, "
            f"object_id={request.object_id}"
        )

        try:
            # Fetch from store
            res = message_handler.pull_object(
                request=request,
                state=self.state_factory.state(),
            )
        except InvalidRunStatusException as e:
            raise FlowerError(
                ApiErrorCode.FLEET_RUN_STATUS_NOT_ALLOWED,
                f"{error_context}, exception: {e.message}",
                public_details=f"Object_id: {request.object_id}",
            ) from e

        return res

    def ConfirmMessageReceived(
        self, request: ConfirmMessageReceivedRequest, context: grpc.ServicerContext
    ) -> ConfirmMessageReceivedResponse:
        """Confirm message received."""
        log(
            DEBUG,
            "[Fleet.ConfirmMessageReceived] Message with ID '%s' has been received",
            request.message_object_id,
        )

        try:
            res = message_handler.confirm_message_received(
                request=request,
                state=self.state_factory.state(),
                store=self.objectstore_factory.store(),
            )
        except InvalidRunStatusException as e:
            raise FlowerError(
                ApiErrorCode.FLEET_RUN_STATUS_NOT_ALLOWED,
                f"SuperNode {request.node.node_id}, exception: {e.message}",
            ) from e

        return res
