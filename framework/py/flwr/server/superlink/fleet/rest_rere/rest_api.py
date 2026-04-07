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
"""Experimental REST API server."""


from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar, cast

from google.protobuf.message import Message as GrpcMessage

from flwr.common.exit import ExitCode, flwr_exit
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
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory
from flwr.supercore.object_store import ObjectStore, ObjectStoreFactory

try:
    from starlette.applications import Starlette
    from starlette.datastructures import Headers
    from starlette.exceptions import HTTPException
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.routing import Route
except ModuleNotFoundError:
    flwr_exit(ExitCode.COMMON_MISSING_EXTRA_REST)


GrpcRequest = TypeVar("GrpcRequest", bound=GrpcMessage)
GrpcResponse = TypeVar("GrpcResponse", bound=GrpcMessage)

GrpcAsyncFunction = Callable[[GrpcRequest], Awaitable[GrpcResponse]]
RestEndPoint = Callable[[Request], Awaitable[Response]]

routes = []


def rest_request_response(
    grpc_request_type: type[GrpcRequest],
) -> Callable[[GrpcAsyncFunction[GrpcRequest, GrpcResponse]], RestEndPoint]:
    """Convert an async gRPC-based function into a RESTful HTTP endpoint."""

    def decorator(func: GrpcAsyncFunction[GrpcRequest, GrpcResponse]) -> RestEndPoint:

        async def wrapper(request: Request) -> Response:
            _check_headers(request.headers)

            # Get the request body as raw bytes
            grpc_req_bytes: bytes = await request.body()

            # Deserialize ProtoBuf
            grpc_req = grpc_request_type.FromString(grpc_req_bytes)
            grpc_res = await func(grpc_req)
            return Response(
                status_code=200,
                content=grpc_res.SerializeToString(),
                headers={"Content-Type": "application/protobuf"},
            )

        # Register route
        path = f"/api/v0/fleet/{func.__name__.replace('_', '-')}"
        routes.append(Route(path, wrapper, methods=["POST"]))
        return wrapper

    return decorator


@rest_request_response(RegisterNodeFleetRequest)
async def register_node(
    request: RegisterNodeFleetRequest,
) -> RegisterNodeFleetResponse:
    """Register a node (Fleet API only)."""
    # Get state from app
    state: LinkState = cast(LinkStateFactory, app.state.STATE_FACTORY).state()

    # Handle message
    return message_handler.register_node(request=request, state=state)


@rest_request_response(ActivateNodeRequest)
async def activate_node(
    request: ActivateNodeRequest,
) -> ActivateNodeResponse:
    """Activate a node."""
    # Get state from app
    state: LinkState = cast(LinkStateFactory, app.state.STATE_FACTORY).state()

    # Handle message
    return message_handler.activate_node(request=request, state=state)


@rest_request_response(DeactivateNodeRequest)
async def deactivate_node(
    request: DeactivateNodeRequest,
) -> DeactivateNodeResponse:
    """Deactivate a node."""
    # Get state from app
    state: LinkState = cast(LinkStateFactory, app.state.STATE_FACTORY).state()

    # Handle message
    return message_handler.deactivate_node(request=request, state=state)


@rest_request_response(UnregisterNodeFleetRequest)
async def unregister_node(
    request: UnregisterNodeFleetRequest,
) -> UnregisterNodeFleetResponse:
    """Unregister a node (Fleet API only)."""
    # Get state from app
    state: LinkState = cast(LinkStateFactory, app.state.STATE_FACTORY).state()

    # Handle message
    return message_handler.unregister_node(request=request, state=state)


@rest_request_response(PullMessagesRequest)
async def pull_messages(request: PullMessagesRequest) -> PullMessagesResponse:
    """Pull PullMessages."""
    # Get state from app
    state: LinkState = cast(LinkStateFactory, app.state.STATE_FACTORY).state()
    store: ObjectStore = cast(ObjectStoreFactory, app.state.OBJECTSTORE_FACTORY).store()

    # Handle message
    return message_handler.pull_messages(request=request, state=state, store=store)


@rest_request_response(PushMessagesRequest)
async def push_messages(request: PushMessagesRequest) -> PushMessagesResponse:
    """Pull PushMessages."""
    # Get state from app
    state: LinkState = cast(LinkStateFactory, app.state.STATE_FACTORY).state()
    store: ObjectStore = cast(ObjectStoreFactory, app.state.OBJECTSTORE_FACTORY).store()

    # Handle message
    return message_handler.push_messages(request=request, state=state, store=store)


@rest_request_response(PullObjectRequest)
async def pull_object(request: PullObjectRequest) -> PullObjectResponse:
    """Pull PullObject."""
    # Get state from app
    state: LinkState = cast(LinkStateFactory, app.state.STATE_FACTORY).state()
    store: ObjectStore = cast(ObjectStoreFactory, app.state.OBJECTSTORE_FACTORY).store()

    # Handle message
    return message_handler.pull_object(request=request, state=state, store=store)


@rest_request_response(PushObjectRequest)
async def push_object(request: PushObjectRequest) -> PushObjectResponse:
    """Pull PushObject."""
    # Get state from app
    state: LinkState = cast(LinkStateFactory, app.state.STATE_FACTORY).state()
    store: ObjectStore = cast(ObjectStoreFactory, app.state.OBJECTSTORE_FACTORY).store()

    # Handle message
    return message_handler.push_object(request=request, state=state, store=store)


@rest_request_response(SendNodeHeartbeatRequest)
async def send_node_heartbeat(
    request: SendNodeHeartbeatRequest,
) -> SendNodeHeartbeatResponse:
    """Send node heartbeat."""
    # Get state from app
    state: LinkState = cast(LinkStateFactory, app.state.STATE_FACTORY).state()

    # Handle message
    return message_handler.send_node_heartbeat(request=request, state=state)


@rest_request_response(GetRunRequest)
async def get_run(request: GetRunRequest) -> GetRunResponse:
    """GetRun."""
    # Get state from app
    state: LinkState = cast(LinkStateFactory, app.state.STATE_FACTORY).state()
    store: ObjectStore = cast(ObjectStoreFactory, app.state.OBJECTSTORE_FACTORY).store()

    # Handle message
    return message_handler.get_run(request=request, state=state, store=store)


@rest_request_response(GetFabRequest)
async def get_fab(request: GetFabRequest) -> GetFabResponse:
    """GetFab."""
    # Get state from app
    state: LinkState = cast(LinkStateFactory, app.state.STATE_FACTORY).state()
    store: ObjectStore = cast(ObjectStoreFactory, app.state.OBJECTSTORE_FACTORY).store()

    # Handle message
    return message_handler.get_fab(request=request, state=state, store=store)


@rest_request_response(ConfirmMessageReceivedRequest)
async def confirm_message_received(
    request: ConfirmMessageReceivedRequest,
) -> ConfirmMessageReceivedResponse:
    """Confirm message received."""
    # Get state from app
    state: LinkState = cast(LinkStateFactory, app.state.STATE_FACTORY).state()
    store: ObjectStore = cast(ObjectStoreFactory, app.state.OBJECTSTORE_FACTORY).store()

    # Handle message
    return message_handler.confirm_message_received(
        request=request, state=state, store=store
    )


app: Starlette = Starlette(
    debug=False,
    routes=routes,
)


def _check_headers(headers: Headers) -> None:
    """Check if expected headers are set."""
    if "content-type" not in headers:
        raise HTTPException(status_code=400, detail="Missing header `Content-Type`")
    if headers["content-type"] != "application/protobuf":
        raise HTTPException(status_code=400, detail="Unsupported `Content-Type`")
    if "accept" not in headers:
        raise HTTPException(status_code=400, detail="Missing header `Accept`")
    if headers["accept"] != "application/protobuf":
        raise HTTPException(status_code=400, detail="Unsupported `Accept`")
