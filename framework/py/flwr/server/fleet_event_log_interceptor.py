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
"""Flower Fleet API event log interceptor."""


from collections.abc import Callable
from typing import Any, cast

import grpc
from google.protobuf.message import Message as GrpcMessage

from flwr.common.event_log_plugin.event_log_plugin import EventLogWriterPlugin
from flwr.supercore.event_log.typing import LogEntry


class FleetEventLogInterceptor(grpc.ServerInterceptor):  # type: ignore
    """Fleet API interceptor for logging events."""

    def __init__(self, log_plugin: EventLogWriterPlugin) -> None:
        self.log_plugin = log_plugin

    def intercept_service(
        self,
        continuation: Callable[[Any], Any],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """Flower Fleet API server interceptor logging logic.

        Intercept all unary-unary calls from users and log the event. Continue RPC call
        if event logger is enabled on the SuperLink, else, terminate RPC call by setting
        context to abort.
        """
        # Only apply to Fleet service
        if not handler_call_details.method.startswith("/flwr.proto.Fleet/"):
            return continuation(handler_call_details)

        # One of the method handlers in
        # `flwr.server.superlink.fleet.grpc_rere.fleet_servicer.FleetServicer`
        method_handler: grpc.RpcMethodHandler = continuation(handler_call_details)
        method_name: str = handler_call_details.method
        return self._generic_event_log_unary_method_handler(method_handler, method_name)

    def _generic_event_log_unary_method_handler(
        self, method_handler: grpc.RpcMethodHandler, method_name: str
    ) -> grpc.RpcMethodHandler:
        def _generic_method_handler(
            request: GrpcMessage,
            context: grpc.ServicerContext,
        ) -> GrpcMessage:
            log_entry: LogEntry
            # Log before call
            log_entry = self.log_plugin.compose_log_before_event(
                request=request,
                context=context,
                account_info=None,
                method_name=method_name,
            )
            self.log_plugin.write_log(log_entry)

            call = method_handler.unary_unary
            unary_response, error = None, None
            try:
                unary_response = cast(GrpcMessage, call(request, context))
            except BaseException as e:
                error = e
                raise
            finally:
                log_entry = self.log_plugin.compose_log_after_event(
                    request=request,
                    context=context,
                    account_info=None,
                    method_name=method_name,
                    response=unary_response or error,
                )
                self.log_plugin.write_log(log_entry)
            return unary_response

        if method_handler.unary_unary:
            message_handler = grpc.unary_unary_rpc_method_handler
        else:
            # If the method type is not `unary_unary` raise an error
            raise NotImplementedError("This RPC method type is not supported.")
        return message_handler(
            _generic_method_handler,
            request_deserializer=method_handler.request_deserializer,
            response_serializer=method_handler.response_serializer,
        )
