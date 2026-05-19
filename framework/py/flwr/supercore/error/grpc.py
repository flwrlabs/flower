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
"""GRPC-specific translation utilities for Flower API errors."""


from collections.abc import Iterator
from contextlib import contextmanager
from logging import ERROR

import grpc
from grpc import StatusCode

from flwr.common.logger import log

from .base import FlowerError
from .catalog import API_ERROR_MAP

INTERNAL_SERVER_ERROR_MESSAGE = "Internal server error."


@contextmanager
def rpc_error_translator(
    context: grpc.ServicerContext, rpc_name: str
) -> Iterator[None]:
    """Translate FlowerError into a sanitized gRPC error."""
    try:
        yield
    except FlowerError as err:
        try:
            error_spec = API_ERROR_MAP[err.code]
            grpc_status = error_spec.status_code
            public_message = error_spec.public_message
        except (ValueError, KeyError):
            grpc_status = StatusCode.INTERNAL
            public_message = INTERNAL_SERVER_ERROR_MESSAGE

        # Log error as is
        msg = f"[{rpc_name}][ApiError:{err.code}] {err.message}"
        log(ERROR, msg)
        # Return sanitized error to client
        context.abort(grpc_status, err.to_json(public_message))
        raise grpc.RpcError() from None  # Unreachable, but satisfies type checker
    except Exception as err:
        # Let pass through if `context.abort()` is called
        if _is_aborted(context, err):
            raise

        # Log unexpected exceptions and translate into INTERNAL
        msg = f"[{rpc_name}][UnexpectedError:{type(err).__name__}] {err}"
        log(ERROR, msg)
        context.abort(StatusCode.INTERNAL, INTERNAL_SERVER_ERROR_MESSAGE)
        raise grpc.RpcError() from None  # Unreachable, but satisfies type checker


def _is_aborted(context: grpc.ServicerContext, err: Exception) -> bool:
    """Check if the gRPC context has been aborted."""
    # Check `.code()` first (experimental API in grpcio==1.78.1)
    if hasattr(context, "code") and context.code() not in (None, StatusCode.OK):
        return True
    # Fallback: check if the error is `Exception` with no message
    if err.__class__ is Exception and not str(err):
        return True
    return False
