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
"""gRPC server interceptor for Flower API error translation."""


from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import grpc
from google.protobuf.message import Message as GrpcMessage

from flwr.supercore.error import rpc_error_translator


class RpcErrorTranslationServerInterceptor(grpc.ServerInterceptor):  # type: ignore
    """Translate Flower API errors raised while handling server RPCs."""

    def intercept_service(
        self,
        continuation: Callable[[Any], Any],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """Wrap unary server RPC handlers with rpc_error_translator."""
        method_handler: grpc.RpcMethodHandler = continuation(handler_call_details)
        if method_handler is None:
            return method_handler

        rpc_name = handler_call_details.method

        unary_unary_handler = method_handler.unary_unary
        if unary_unary_handler is not None:

            def unary_unary(
                request: GrpcMessage, context: grpc.ServicerContext
            ) -> GrpcMessage:
                with rpc_error_translator(context, rpc_name):
                    return unary_unary_handler(request, context)  # type: ignore

            return grpc.unary_unary_rpc_method_handler(
                unary_unary,
                request_deserializer=method_handler.request_deserializer,
                response_serializer=method_handler.response_serializer,
            )

        unary_stream_handler = method_handler.unary_stream
        if unary_stream_handler is not None:

            def unary_stream(
                request: GrpcMessage, context: grpc.ServicerContext
            ) -> Iterator[GrpcMessage]:
                with rpc_error_translator(context, rpc_name):
                    yield from unary_stream_handler(request, context)

            return grpc.unary_stream_rpc_method_handler(
                unary_stream,
                request_deserializer=method_handler.request_deserializer,
                response_serializer=method_handler.response_serializer,
            )

        return method_handler
