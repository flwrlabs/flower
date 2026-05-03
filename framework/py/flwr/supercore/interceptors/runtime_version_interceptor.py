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
"""Runtime version metadata interceptors."""


from __future__ import annotations

from collections.abc import Callable
from logging import WARN
from typing import Any

import grpc
from google.protobuf.message import Message as GrpcMessage

from flwr.common.logger import log
from flwr.supercore.constant import VERSION_INCOMPATIBILITY_MESSAGE_METADATA_KEY
from flwr.supercore.runtime_version_compatibility import RuntimeVersionMetadata
from flwr.supercore.utils import get_metadata_str


class RuntimeVersionClientInterceptor(
    grpc.UnaryUnaryClientInterceptor,  # type: ignore[misc]
    grpc.UnaryStreamClientInterceptor,  # type: ignore[misc]
):
    """Attach Flower runtime version metadata to outbound unary-unary and unary-stream
    RPCs."""

    def __init__(self, component_name: str) -> None:
        self._metadata = RuntimeVersionMetadata.from_local_component(component_name)
        self._compatibility_warning_logged = False

    def _maybe_log_incompat_warning(
        self,
        grpc_metadata: Any | None,
    ) -> None:
        if self._compatibility_warning_logged:
            return

        incompat_message = get_metadata_str(
            grpc_metadata,
            VERSION_INCOMPATIBILITY_MESSAGE_METADATA_KEY,
        )
        if incompat_message:
            self._compatibility_warning_logged = True
            log(WARN, incompat_message)

    def intercept_unary_unary(
        self,
        continuation: Callable[[Any, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request: GrpcMessage,
    ) -> grpc.Call:
        """Add the runtime version metadata headers."""
        details = client_call_details._replace(
            metadata=self._metadata.append_to_grpc_metadata(
                client_call_details.metadata
            )
        )
        call: grpc.Call = continuation(details, request)

        # Log the incompatibility message from the response metadata
        self._maybe_log_incompat_warning(call.trailing_metadata())

        return call

    def intercept_unary_stream(
        self,
        continuation: Callable[[Any, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request: GrpcMessage,
    ) -> grpc.Call:
        """Add the runtime version metadata headers for unary-stream RPCs."""
        details = client_call_details._replace(
            metadata=self._metadata.append_to_grpc_metadata(
                client_call_details.metadata
            )
        )
        call: grpc.Call = continuation(details, request)

        def _log_incompat_warning() -> None:
            self._maybe_log_incompat_warning(call.trailing_metadata())

        if not call.add_callback(_log_incompat_warning):
            _log_incompat_warning()

        return call


class RuntimeVersionServerInterceptor(grpc.ServerInterceptor):  # type: ignore[misc]
    """Observe Flower runtime version metadata on inbound unary RPCs."""

    def __init__(
        self,
        *,
        connection_name: str,
        local_metadata: RuntimeVersionMetadata,
    ) -> None:
        self._connection_name = connection_name
        self._local_metadata = local_metadata

    def intercept_service(
        self,
        continuation: Callable[[Any], Any],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """Parse peer runtime metadata, then continue normal RPC handling."""
        method_handler: grpc.RpcMethodHandler = continuation(handler_call_details)
        if method_handler is None:
            return method_handler

        # Parse and validate peer metadata
        peer_metadata, incompat_details = RuntimeVersionMetadata.from_grpc_metadata(
            handler_call_details.invocation_metadata
        )

        # Check compatibility and return any rejection message
        if incompat_details is None:
            incompat_details = self._local_metadata.check_compatibility(peer_metadata)

        # Prepare trailing metadata
        trailing_metadata: tuple[tuple[str, str], ...] = ()
        if incompat_details:
            incompat_message = (
                "Runtime version compatibility check failed for "
                f"{self._connection_name}. {incompat_details}"
            )
            trailing_metadata += (
                (VERSION_INCOMPATIBILITY_MESSAGE_METADATA_KEY, incompat_message),
            )

        def maybe_set_trailing_metadata(
            context: grpc.ServicerContext,
        ) -> None:
            if trailing_metadata:
                context.set_trailing_metadata(trailing_metadata)

        if method_handler.unary_unary is not None:

            def wrapped(
                request: GrpcMessage, context: grpc.ServicerContext
            ) -> GrpcMessage:
                maybe_set_trailing_metadata(context)
                return method_handler.unary_unary(request, context)  # type: ignore

            return grpc.unary_unary_rpc_method_handler(
                wrapped,
                request_deserializer=method_handler.request_deserializer,
                response_serializer=method_handler.response_serializer,
            )

        if method_handler.unary_stream is not None:

            def wrapped_stream(
                request: GrpcMessage, context: grpc.ServicerContext
            ) -> Any:
                maybe_set_trailing_metadata(context)
                yield from method_handler.unary_stream(request, context)

            return grpc.unary_stream_rpc_method_handler(
                wrapped_stream,
                request_deserializer=method_handler.request_deserializer,
                response_serializer=method_handler.response_serializer,
            )

        return method_handler


def create_serverappio_runtime_version_server_interceptor(
    connection_name: str = "Caller <-> SuperLink ServerAppIo API",
) -> RuntimeVersionServerInterceptor:
    """Create the default runtime version interceptor for ServerAppIo."""
    return RuntimeVersionServerInterceptor(
        connection_name=connection_name,
        local_metadata=RuntimeVersionMetadata.from_local_component("SuperLink"),
    )
