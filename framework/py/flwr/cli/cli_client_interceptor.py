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
# ===============================================================================
"""Client interceptors for Flower CLI metadata."""

from collections.abc import Callable
from typing import Any

import grpc
import httpx

from flwr.supercore.constant import FLWR_CLIENT_METADATA_KEY
from flwr.supercore.protobuf.client import ProtobufCall, ProtobufRequestContext


class CliClientInterceptor(
    grpc.UnaryUnaryClientInterceptor,  # type: ignore[misc]
    grpc.UnaryStreamClientInterceptor,  # type: ignore[misc]
):
    """Attach the CLI client identifier to every outgoing Control API call."""

    def _intercept_call(
        self,
        continuation: Callable[[Any, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request: Any,
    ) -> grpc.Call:
        """Add the CLI client identifier to a gRPC call."""
        metadata = [
            (key, value)
            for key, value in (client_call_details.metadata or [])
            if key != FLWR_CLIENT_METADATA_KEY
        ]
        metadata.append((FLWR_CLIENT_METADATA_KEY, "cli"))
        details = client_call_details._replace(metadata=metadata)
        return continuation(details, request)

    def intercept_unary_unary(
        self,
        continuation: Callable[[Any, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request: Any,
    ) -> grpc.Call:
        """Add the CLI client identifier to a unary-unary call."""
        return self._intercept_call(continuation, client_call_details, request)

    def intercept_unary_stream(
        self,
        continuation: Callable[[Any, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request: Any,
    ) -> grpc.Call:
        """Add the CLI client identifier to a unary-stream call."""
        return self._intercept_call(continuation, client_call_details, request)


class CliClientHttpInterceptor:
    """Attach the CLI client identifier to protobuf-over-HTTP requests."""

    def intercept(
        self,
        context: ProtobufRequestContext,
        call_next: ProtobufCall,
    ) -> httpx.Response:
        """Add the CLI client identifier to an HTTP call."""
        context.request.headers[FLWR_CLIENT_METADATA_KEY] = "cli"
        return call_next(context)
