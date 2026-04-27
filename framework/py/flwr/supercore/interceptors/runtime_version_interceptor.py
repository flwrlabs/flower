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
from typing import Any, cast

import grpc
from google.protobuf.message import Message as GrpcMessage

from flwr.supercore.runtime_version_compatibility import (
    CompatibilityResult,
    RuntimeVersionMetadata,
    format_incompatible_version_message,
    format_invalid_metadata_message,
)


class RuntimeVersionClientInterceptor(grpc.UnaryUnaryClientInterceptor):  # type: ignore
    """Attach Flower runtime version metadata to outbound unary RPCs."""

    def __init__(self, component_name: str) -> None:
        self._metadata = RuntimeVersionMetadata.from_local_component(component_name)

    def intercept_unary_unary(
        self,
        continuation: Callable[[Any, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request: GrpcMessage,
    ) -> grpc.Call:
        """Add or replace the runtime version metadata headers."""
        details = client_call_details._replace(
            metadata=self._metadata.append_to_grpc_metadata(
                client_call_details.metadata
            )
        )
        return continuation(details, request)


class RuntimeVersionServerInterceptor(grpc.ServerInterceptor):  # type: ignore
    """Validate Flower runtime version metadata on inbound unary RPCs."""

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
        """Reject explicit invalid or incompatible peer metadata before handling."""
        method_handler = continuation(handler_call_details)
        if method_handler is None or method_handler.unary_unary is None:
            return method_handler

        peer_metadata, metadata_error = RuntimeVersionMetadata.from_grpc_metadata(
            handler_call_details.invocation_metadata
        )
        if metadata_error is not None:
            compatibility = CompatibilityResult(
                status="invalid",
                reason=metadata_error,
                local_metadata=self._local_metadata,
                peer_metadata=peer_metadata,
                local_version=None,
                peer_version=None,
            )
        else:
            compatibility = self._local_metadata.check_compatibility_with(peer_metadata)
        if compatibility.status in {"missing", "disabled", "compatible"}:
            return method_handler

        unary_handler = cast(
            Callable[[GrpcMessage, grpc.ServicerContext], GrpcMessage],
            method_handler.unary_unary,
        )

        def _version_checked_handler(
            request: GrpcMessage,
            context: grpc.ServicerContext,
        ) -> GrpcMessage:
            if compatibility.status == "invalid":
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    format_invalid_metadata_message(
                        self._connection_name,
                        compatibility.reason or "Unknown metadata error.",
                    ),
                )
                raise RuntimeError("Should not reach this point")
            if compatibility.status == "incompatible":
                peer_metadata = compatibility.peer_metadata
                if peer_metadata is None:
                    context.abort(
                        grpc.StatusCode.FAILED_PRECONDITION,
                        format_invalid_metadata_message(
                            self._connection_name,
                            "Peer metadata is unavailable for incompatibility check.",
                        ),
                    )
                    raise RuntimeError("Should not reach this point")
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    format_incompatible_version_message(
                        self._connection_name,
                        self._local_metadata,
                        peer_metadata,
                    ),
                )
                raise RuntimeError("Should not reach this point")
            return unary_handler(request, context)

        return grpc.unary_unary_rpc_method_handler(
            _version_checked_handler,
            request_deserializer=method_handler.request_deserializer,
            response_serializer=method_handler.response_serializer,
        )


def create_serverappio_runtime_version_server_interceptor(
    connection_name: str = "Caller <-> SuperLink ServerAppIo API",
) -> RuntimeVersionServerInterceptor:
    """Create the default runtime version interceptor for ServerAppIo."""
    return RuntimeVersionServerInterceptor(
        connection_name=connection_name,
        local_metadata=RuntimeVersionMetadata.from_local_component("superlink"),
    )
