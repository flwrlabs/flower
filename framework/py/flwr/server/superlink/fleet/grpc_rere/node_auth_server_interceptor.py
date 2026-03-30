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
"""Flower server interceptor."""


import datetime
from collections.abc import Callable
from typing import Any, cast

import grpc
from google.protobuf.message import Message as GrpcMessage

from flwr.common import now
from flwr.common.constant import (
    PUBLIC_KEY_HEADER,
    SIGNATURE_HEADER,
    SYSTEM_TIME_TOLERANCE,
    TIMESTAMP_HEADER,
    TIMESTAMP_TOLERANCE,
)
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    ActivateNodeRequest,
    RegisterNodeFleetRequest,
)
from flwr.server.superlink.linkstate import LinkStateFactory
from flwr.supercore.primitives.asymmetric import bytes_to_public_key, verify_signature

MIN_TIMESTAMP_DIFF = -SYSTEM_TIME_TOLERANCE
MAX_TIMESTAMP_DIFF = TIMESTAMP_TOLERANCE + SYSTEM_TIME_TOLERANCE


def _unary_unary_rpc_terminator(
    message: str, code: Any = grpc.StatusCode.UNAUTHENTICATED
) -> grpc.RpcMethodHandler:
    def terminate(_request: GrpcMessage, context: grpc.ServicerContext) -> GrpcMessage:
        context.abort(code, message)
        raise RuntimeError("Should not reach this point")  # Make mypy happy

    return grpc.unary_unary_rpc_method_handler(terminate)


class NodeAuthServerInterceptor(grpc.ServerInterceptor):  # type: ignore
    """Server interceptor for node authentication.

    Parameters
    ----------
    state_factory : LinkStateFactory
        A factory for creating new instances of LinkState.
    """

    def __init__(self, state_factory: LinkStateFactory):
        self.state_factory = state_factory

    def intercept_service(  # pylint: disable=too-many-return-statements
        self,
        continuation: Callable[[Any], Any],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """Flower server interceptor authentication logic.

        Intercept all unary calls from nodes and authenticate nodes by validating auth
        metadata sent by the node. Continue RPC call if node is authenticated, else,
        terminate RPC call by setting context to abort.
        """
        # Only apply to Fleet service
        if not handler_call_details.method.startswith("/flwr.proto.Fleet/"):
            return continuation(handler_call_details)

        metadata_dict = dict(handler_call_details.invocation_metadata)

        # Retrieve info from the metadata
        try:
            node_pk_bytes = cast(bytes, metadata_dict[PUBLIC_KEY_HEADER])
            timestamp_iso = cast(str, metadata_dict[TIMESTAMP_HEADER])
            signature = cast(bytes, metadata_dict[SIGNATURE_HEADER])
        except KeyError:
            return _unary_unary_rpc_terminator("Missing authentication metadata")

        # Verify the signature
        node_pk = bytes_to_public_key(node_pk_bytes)
        if not verify_signature(node_pk, timestamp_iso.encode("ascii"), signature):
            return _unary_unary_rpc_terminator("Invalid signature")

        # Verify the timestamp
        current = now()
        time_diff = current - datetime.datetime.fromisoformat(timestamp_iso)
        # Abort the RPC call if the timestamp is too old or in the future
        if not MIN_TIMESTAMP_DIFF < time_diff.total_seconds() < MAX_TIMESTAMP_DIFF:
            return _unary_unary_rpc_terminator("Invalid timestamp")

        # Continue the RPC call: One of the method handlers in
        # `flwr.server.superlink.fleet.grpc_rere.fleet_server.FleetServicer`
        method_handler: grpc.RpcMethodHandler = continuation(handler_call_details)
        return self._wrap_method_handler(method_handler, node_pk_bytes)

    def _wrap_method_handler(
        self,
        method_handler: grpc.RpcMethodHandler,
        received_public_key: bytes,
    ) -> grpc.RpcMethodHandler:
        def _generic_method_handler(
            request: GrpcMessage,
            context: grpc.ServicerContext,
        ) -> GrpcMessage:
            # Note: This function runs in a different thread
            # than the `intercept_service` function.

            # Skip registration and activation requests
            if not isinstance(request, (RegisterNodeFleetRequest, ActivateNodeRequest)):
                # Retrieve the node ID from the request
                if hasattr(request, "node"):
                    received_node_id = request.node.node_id
                else:
                    received_node_id = request.node_id  # type: ignore[attr-defined]

                # Get the actual node ID based on the received public key
                node_id = self.state_factory.state().get_node_id_by_public_key(
                    received_public_key
                )

                # Verify that the received node ID matches the actual node ID
                if received_node_id != node_id:
                    context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid node ID")

            response: GrpcMessage = method_handler.unary_unary(request, context)
            return response

        return grpc.unary_unary_rpc_method_handler(
            _generic_method_handler,
            request_deserializer=method_handler.request_deserializer,
            response_serializer=method_handler.response_serializer,
        )
