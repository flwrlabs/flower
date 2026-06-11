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
"""Flower Control API event log interceptor."""


from collections.abc import Callable, Iterator
from typing import Any, cast

import grpc
from google.protobuf.message import Message as GrpcMessage

from flwr.common.event_log_plugin.event_log_plugin import EventLogWriterPlugin
from flwr.supercore.event_log.typing import LogEntry

from .control_account_auth_interceptor import get_current_account_info


class ControlEventLogInterceptor(grpc.ServerInterceptor):  # type: ignore
    """Control API interceptor for logging events."""

    def __init__(self, log_plugin: EventLogWriterPlugin) -> None:
        self.log_plugin = log_plugin

    def intercept_service(
        self,
        continuation: Callable[[Any], Any],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """Flower server interceptor logging logic.

        Intercept all unary-unary/unary-stream calls from users and log the event.
        Continue RPC call if event logger is enabled on the SuperLink, else, terminate
        RPC call by setting context to abort.
        """
        # Only apply to Control service
        if not handler_call_details.method.startswith("/flwr.proto.Control/"):
            return continuation(handler_call_details)

        # One of the method handlers in
        # `flwr.superlink.servicer.control.ControlServicer`
        method_handler: grpc.RpcMethodHandler = continuation(handler_call_details)
        method_name: str = handler_call_details.method
        return self._generic_event_log_unary_method_handler(method_handler, method_name)

    def _generic_event_log_unary_method_handler(
        self, method_handler: grpc.RpcMethodHandler, method_name: str
    ) -> grpc.RpcMethodHandler:
        def _generic_method_handler(
            request: GrpcMessage,
            context: grpc.ServicerContext,
        ) -> GrpcMessage | Iterator[GrpcMessage] | BaseException:
            log_entry: LogEntry
            # Log before call
            log_entry = self.log_plugin.compose_log_before_event(
                request=request,
                context=context,
                account_info=get_current_account_info(),
                method_name=method_name,
            )
            self.log_plugin.write_log(log_entry)

            # For unary-unary calls, log after the call immediately
            if method_handler.unary_unary:
                unary_response, error = None, None
                try:
                    unary_response = cast(
                        GrpcMessage, method_handler.unary_unary(request, context)
                    )
                except BaseException as e:
                    error = e
                    raise
                finally:
                    log_entry = self.log_plugin.compose_log_after_event(
                        request=request,
                        context=context,
                        account_info=get_current_account_info(),
                        method_name=method_name,
                        response=unary_response or error,
                    )
                    self.log_plugin.write_log(log_entry)
                return unary_response

            # For unary-stream calls, wrap the response iterator and write the event log
            # after iteration completes
            if method_handler.unary_stream:
                response_iterator = cast(
                    Iterator[GrpcMessage],
                    method_handler.unary_stream(request, context),
                )

                def response_wrapper() -> Iterator[GrpcMessage]:
                    stream_response, error = None, None
                    try:
                        # pylint: disable=use-yield-from
                        for stream_response in response_iterator:
                            yield stream_response
                    except BaseException as e:
                        error = e
                        raise
                    finally:
                        # This block is executed after the client has consumed
                        # the entire stream, or if iteration is interrupted
                        log_entry = self.log_plugin.compose_log_after_event(
                            request=request,
                            context=context,
                            account_info=get_current_account_info(),
                            method_name=method_name,
                            response=stream_response or error,
                        )
                        self.log_plugin.write_log(log_entry)

                return response_wrapper()

            raise RuntimeError()  # This line is unreachable

        if method_handler.unary_unary:
            message_handler = grpc.unary_unary_rpc_method_handler
        elif method_handler.unary_stream:
            message_handler = grpc.unary_stream_rpc_method_handler
        else:
            # If the method type is not `unary_unary` or `unary_stream`, raise an error
            raise NotImplementedError("This RPC method type is not supported.")
        return message_handler(
            _generic_method_handler,
            request_deserializer=method_handler.request_deserializer,
            response_serializer=method_handler.response_serializer,
        )
