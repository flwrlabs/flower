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
"""Tests for gRPC error translation utilities."""


from unittest.mock import Mock

import grpc
import pytest

from .base import ApiErrorCode, FlowerError
from .catalog import API_ERROR_MAP
from .grpc import INTERNAL_SERVER_ERROR_MESSAGE, rpc_error_translator


def test_rpc_error_translator_mapped_flower_error() -> None:
    """Translate a mapped FlowerError into its configured gRPC contract."""
    context = Mock(spec=grpc.ServicerContext)
    context.abort.side_effect = grpc.RpcError()
    context.code.return_value = None

    with pytest.raises(grpc.RpcError):
        with rpc_error_translator(context, "MockApi.MockRpc"):
            raise FlowerError(
                ApiErrorCode.NO_FEDERATION_MANAGEMENT_SUPPORT,
                "internal diagnostic message",
            )

    spec = API_ERROR_MAP[ApiErrorCode.NO_FEDERATION_MANAGEMENT_SUPPORT]
    context.abort.assert_called_once_with(spec.status_code, spec.public_message)


def test_rpc_error_translator_grpc_error() -> None:
    """Allow `context.abort()` to propagate unmodified."""
    context = Mock(spec=grpc.ServicerContext)
    context.code.return_value = grpc.StatusCode.UNKNOWN

    with pytest.raises(Exception) as err:  # noqa: B017
        with rpc_error_translator(context, "MockApi.MockRpc"):
            raise Exception  # Same as `context.abort()`  # pylint: disable=W0719

    assert err.value.__class__ is Exception
    context.code.assert_called_once()
    context.abort.assert_not_called()


def test_rpc_error_translator_unmapped_flower_error() -> None:
    """Translate an unmapped FlowerError into INTERNAL."""
    context = Mock(spec=grpc.ServicerContext)
    context.abort.side_effect = grpc.RpcError()
    context.code.return_value = None

    with pytest.raises(grpc.RpcError):
        with rpc_error_translator(context, "MockApi.MockRpc"):
            raise FlowerError(999, "internal diagnostic message")

    context.abort.assert_called_once_with(
        grpc.StatusCode.INTERNAL,
        INTERNAL_SERVER_ERROR_MESSAGE,
    )


def test_rpc_error_translator_unexpected_error() -> None:
    """Translate unexpected errors into INTERNAL."""
    context = Mock(spec=grpc.ServicerContext)
    context.abort.side_effect = grpc.RpcError()
    context.code.return_value = None

    with pytest.raises(grpc.RpcError):
        with rpc_error_translator(context, "MockApi.MockRpc"):
            raise RuntimeError("unexpected failure")

    context.abort.assert_called_once_with(
        grpc.StatusCode.INTERNAL,
        INTERNAL_SERVER_ERROR_MESSAGE,
    )
