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
"""Tests for FlowerError client-side gRPC formatting."""


from typing import cast
from unittest.mock import Mock

import grpc
import pytest

from flwr.proto.run_pb2 import GetRunRequest  # pylint: disable=E0611
from flwr.supercore.error import ApiErrorCode, FlowerError

from .rpc_error_translation_client_interceptor import (
    RpcErrorTranslationClientInterceptor,
)


def _make_rpc_error(details: str) -> grpc.RpcError:
    error = grpc.RpcError()
    error.details = Mock(return_value=details)
    return error


def _intercept_with_error(rpc_error: grpc.RpcError) -> None:
    def continuation(
        _client_call_details: grpc.ClientCallDetails,
        _request: GetRunRequest,
    ) -> grpc.Call:
        raise rpc_error

    RpcErrorTranslationClientInterceptor().intercept_unary_unary(
        continuation,
        cast(grpc.ClientCallDetails, Mock()),
        GetRunRequest(run_id=1),
    )


def test_translate_serialized_flower_error() -> None:
    """Serialized FlowerError details should become a readable exception."""
    rpc_error = _make_rpc_error(
        FlowerError(
            ApiErrorCode.FLEET_GET_RUN_FAILED,
            "internal diagnostic message",
            public_details="Run ID not found: 1",
        ).to_json("Failed to get run.")
    )

    with pytest.raises(RuntimeError, match="Failed to get run.\nRun ID not found: 1"):
        _intercept_with_error(rpc_error)


def test_pass_through_non_flower_rpc_error() -> None:
    """Non-Flower gRPC failures should keep their original RpcError type."""
    rpc_error = _make_rpc_error("transport failed")

    with pytest.raises(grpc.RpcError) as err:
        _intercept_with_error(rpc_error)

    assert err.value is rpc_error
