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
"""gRPC client interceptor for formatting Flower API errors."""


from collections.abc import Callable
from typing import Any, cast

import grpc
from google.protobuf.message import Message as GrpcMessage

from flwr.supercore.error import FlowerError, format_flower_error


class RpcErrorTranslationClientInterceptor(grpc.UnaryUnaryClientInterceptor):  # type: ignore
    """Translate serialized FlowerError RPC failures into readable exceptions."""

    def _maybe_raise_flower_error(self, err: grpc.RpcError) -> None:
        details = cast(str | None, err.details())  # pylint: disable=E1101
        if flower_error := FlowerError.from_json(details):
            raise RuntimeError(format_flower_error(flower_error)) from None

    def intercept_unary_unary(
        self,
        continuation: Callable[[Any, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request: GrpcMessage,
    ) -> grpc.Call:
        """Format serialized FlowerError details for unary-unary RPC failures."""
        try:
            call: grpc.Call = continuation(client_call_details, request)
        except grpc.RpcError as err:
            self._maybe_raise_flower_error(err)
            raise

        if isinstance(call, grpc.RpcError):
            self._maybe_raise_flower_error(call)

        return call
