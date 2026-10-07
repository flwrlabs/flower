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
"""Fleet API over protobuf HTTP."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

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
from flwr.server.superlink.linkstate import LinkState
from flwr.supercore.protobuf.routing import ProtobufRoute
from flwr.supercore.protobuf.translation import PROTOBUF_REQUEST_DEPENDENCY
from flwr.superlink.dependencies.linkstate import get_linkstate
from flwr.superlink.servicer.fleet import fleet_handlers

from .node_auth import authenticate_node

router = APIRouter(
    prefix="/v1/fleet",
    tags=["Fleet"],
    route_class=ProtobufRoute,
    dependencies=[Depends(authenticate_node)],
)
LinkStateDependency = Annotated[LinkState, Depends(get_linkstate)]


def get_enable_supernode_auth(request: Request) -> bool:
    """Return whether SuperNode authentication is configured."""
    return bool(request.app.state.enable_supernode_auth)


EnableSuperNodeAuthDependency = Annotated[bool, Depends(get_enable_supernode_auth)]


@router.post("/register-node")
def fleet_register_node(
    request: Annotated[RegisterNodeFleetRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: LinkStateDependency,
    enable_supernode_auth: EnableSuperNodeAuthDependency,
) -> RegisterNodeFleetResponse:
    """Handle Fleet register node."""
    return fleet_handlers.register_node(
        request=request, state=state, enable_supernode_auth=enable_supernode_auth
    )


@router.post("/activate-node")
def fleet_activate_node(
    request: Annotated[ActivateNodeRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: LinkStateDependency,
) -> ActivateNodeResponse:
    """Handle Fleet activate node."""
    return fleet_handlers.activate_node(request=request, state=state)


@router.post("/deactivate-node")
def fleet_deactivate_node(
    request: Annotated[DeactivateNodeRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: LinkStateDependency,
) -> DeactivateNodeResponse:
    """Handle Fleet deactivate node."""
    return fleet_handlers.deactivate_node(request=request, state=state)


@router.post("/unregister-node")
def fleet_unregister_node(
    request: Annotated[UnregisterNodeFleetRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: LinkStateDependency,
    enable_supernode_auth: EnableSuperNodeAuthDependency,
) -> UnregisterNodeFleetResponse:
    """Handle Fleet unregister node."""
    return fleet_handlers.unregister_node(
        request=request, state=state, enable_supernode_auth=enable_supernode_auth
    )


@router.post("/send-node-heartbeat")
def fleet_send_node_heartbeat(
    request: Annotated[SendNodeHeartbeatRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: LinkStateDependency,
) -> SendNodeHeartbeatResponse:
    """Handle Fleet send node heartbeat."""
    return fleet_handlers.send_node_heartbeat(request=request, state=state)


@router.post("/pull-messages")
def fleet_pull_messages(
    request: Annotated[PullMessagesRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: LinkStateDependency,
) -> PullMessagesResponse:
    """Handle Fleet pull messages."""
    return fleet_handlers.pull_messages(
        request=request, state=state, store=state.object_store
    )


@router.post("/push-messages")
def fleet_push_messages(
    request: Annotated[PushMessagesRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: LinkStateDependency,
) -> PushMessagesResponse:
    """Handle Fleet push messages."""
    return fleet_handlers.push_messages(request=request, state=state)


@router.post("/get-run")
def fleet_get_run(
    request: Annotated[GetRunRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: LinkStateDependency,
) -> GetRunResponse:
    """Handle Fleet get run."""
    return fleet_handlers.get_run(request=request, state=state)


@router.post("/get-fab")
def fleet_get_fab(
    request: Annotated[GetFabRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: LinkStateDependency,
) -> GetFabResponse:
    """Handle Fleet get fab."""
    return fleet_handlers.get_fab(request=request, state=state)


@router.post("/push-object")
def fleet_push_object(
    request: Annotated[PushObjectRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: LinkStateDependency,
) -> PushObjectResponse:
    """Handle Fleet push object."""
    return fleet_handlers.push_object(request=request, state=state)


@router.post("/pull-object")
def fleet_pull_object(
    request: Annotated[PullObjectRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: LinkStateDependency,
) -> PullObjectResponse:
    """Handle Fleet pull object."""
    return fleet_handlers.pull_object(request=request, state=state)


@router.post("/confirm-message-received")
def fleet_confirm_message_received(
    request: Annotated[ConfirmMessageReceivedRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: LinkStateDependency,
) -> ConfirmMessageReceivedResponse:
    """Handle Fleet confirm message received."""
    return fleet_handlers.confirm_message_received(
        request=request, state=state, store=state.object_store
    )
