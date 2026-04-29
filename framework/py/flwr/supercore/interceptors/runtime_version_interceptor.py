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
from flwr.supercore.constant import (
    FLWR_COMPONENT_NAME_METADATA_KEY,
    FLWR_PACKAGE_NAME_METADATA_KEY,
    FLWR_PACKAGE_VERSION_METADATA_KEY,
)
from flwr.supercore.runtime_version_compatibility import RuntimeVersionMetadata
from flwr.supercore.utils import find_metadata_keys

_RUNTIME_METADATA_KEYS = (
    FLWR_PACKAGE_NAME_METADATA_KEY,
    FLWR_PACKAGE_VERSION_METADATA_KEY,
    FLWR_COMPONENT_NAME_METADATA_KEY,
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
        existing_runtime_keys = find_metadata_keys(
            client_call_details.metadata,
            _RUNTIME_METADATA_KEYS,
        )
        if existing_runtime_keys:
            log(
                WARN,
                "Outbound gRPC metadata already contains runtime version keys; "
                "replacing existing values for: %s",
                ", ".join(sorted(existing_runtime_keys)),
            )
        details = client_call_details._replace(
            metadata=self._metadata.append_to_grpc_metadata(
                client_call_details.metadata
            )
        )
        return continuation(details, request)


class RuntimeVersionServerInterceptor(grpc.ServerInterceptor):  # type: ignore
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
        method_handler = continuation(handler_call_details)
        if method_handler is None or method_handler.unary_unary is None:
            return method_handler

        RuntimeVersionMetadata.from_grpc_metadata(
            handler_call_details.invocation_metadata
        )
        return method_handler


def create_serverappio_runtime_version_server_interceptor(
    connection_name: str = "Caller <-> SuperLink ServerAppIo API",
) -> RuntimeVersionServerInterceptor:
    """Create the default runtime version interceptor for ServerAppIo."""
    return RuntimeVersionServerInterceptor(
        connection_name=connection_name,
        local_metadata=RuntimeVersionMetadata.from_local_component("superlink"),
    )
