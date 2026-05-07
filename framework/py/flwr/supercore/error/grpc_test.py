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


import json
from unittest.mock import Mock

import grpc
import pytest

from .base import ApiErrorCode, FlowerError
from .catalog import API_ERROR_MAP
from .exceptions import EntitlementError
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
    context.abort.assert_called_once()
    grpc_status, payload = context.abort.call_args.args
    assert grpc_status == spec.status_code
    assert json.loads(payload) == {
        "code": ApiErrorCode.NO_FEDERATION_MANAGEMENT_SUPPORT,
        "public_message": spec.public_message,
        "public_details": None,
    }


def test_flower_error_from_json() -> None:
    """Deserialize the public FlowerError payload on the client side."""
    err = FlowerError(
        ApiErrorCode.RUNTIME_VERSION_INCOMPATIBLE,
        "internal diagnostic message",
        public_details="public details",
    )

    parsed = FlowerError.from_json(err.to_json("public message"))

    assert parsed is not None
    assert parsed.code == ApiErrorCode.RUNTIME_VERSION_INCOMPATIBLE
    assert parsed.message == "public message"
    assert parsed.public_details == "public details"


def test_flower_error_from_json_returns_base_error_for_subclass_call() -> None:
    """Deserialize public payloads into a base FlowerError."""
    err = EntitlementError("internal diagnostic message", entitlement_code=123)

    parsed = EntitlementError.from_json(err.to_json("public message"))

    assert isinstance(parsed, FlowerError)
    assert not isinstance(parsed, EntitlementError)
    assert parsed.code == ApiErrorCode.ENTITLEMENT_ERROR
    assert parsed.message == "public message"
    assert parsed.public_details == "internal diagnostic message"


@pytest.mark.parametrize(
    "value",
    [
        None,
        "not json",
        "[]",
        '{"public_message": "missing code"}',
        '{"code": 1}',
        '{"code": "1", "public_message": "wrong code type"}',
        '{"code": true, "public_message": "wrong code type"}',
        '{"code": 1, "public_message": "ok", "public_details": 1}',
    ],
)
def test_flower_error_from_json_returns_none_for_invalid_payloads(
    value: str | None,
) -> None:
    """Invalid public FlowerError payloads should not raise."""
    assert FlowerError.from_json(value) is None


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
    context.abort.assert_called_once()
    grpc_status, payload = context.abort.call_args.args
    assert grpc_status == grpc.StatusCode.INTERNAL
    assert json.loads(payload) == {
        "code": 999,
        "public_message": INTERNAL_SERVER_ERROR_MESSAGE,
        "public_details": None,
    }


def test_rpc_error_translator_entitlement_error_preserves_error_message() -> None:
    """Keep entitlement details in the translated payload."""
    context = Mock(spec=grpc.ServicerContext)
    context.abort.side_effect = grpc.RpcError()
    context.code.return_value = None

    error_message = "Entitlement check failed: plan does not allow this action."
    entitlement_code = 101

    with pytest.raises(grpc.RpcError):
        with rpc_error_translator(context, "MockApi.MockRpc"):
            raise EntitlementError(error_message, entitlement_code)

    context.abort.assert_called_once()
    grpc_status, payload = context.abort.call_args.args
    assert grpc_status == grpc.StatusCode.PERMISSION_DENIED
    assert json.loads(payload) == {
        "code": ApiErrorCode.ENTITLEMENT_ERROR,
        "public_message": API_ERROR_MAP[ApiErrorCode.ENTITLEMENT_ERROR].public_message,
        "public_details": error_message,
        "entitlement_code": entitlement_code,
    }


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
