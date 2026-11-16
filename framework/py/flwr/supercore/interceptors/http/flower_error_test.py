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
"""Tests for protobuf-over-HTTP Flower error translation."""

from unittest.mock import Mock

import httpx
import pytest

from flwr.proto.runtime_pb2 import GetFileRequest  # pylint: disable=E0611
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.protobuf.client import ProtobufRequestContext

from .flower_error import FlowerErrorHttpInterceptor


@pytest.fixture(name="context")
def request_context() -> ProtobufRequestContext:
    """Create a representative protobuf HTTP request context."""
    return ProtobufRequestContext(
        rpc_method="/flwr.proto.Runtime/GetFile",
        message=GetFileRequest(key="notes.md"),
        request=httpx.Request("POST", "http://runtime.example/v1/runtime/get-file"),
    )


def test_preserves_successful_response(context: ProtobufRequestContext) -> None:
    """Return successful responses unchanged."""
    response = httpx.Response(200, request=context.request)
    call_next = Mock(return_value=response)

    assert FlowerErrorHttpInterceptor().intercept(context, call_next) is response
    call_next.assert_called_once_with(context)


@pytest.mark.parametrize("raised_by_next", [False, True])
def test_translates_flower_error(
    context: ProtobufRequestContext, raised_by_next: bool
) -> None:
    """Translate both returned error responses and raised HTTP status errors."""
    response = httpx.Response(
        409,
        json={
            "code": ApiErrorCode.RUNTIME_FILE_REVISION_CONFLICT,
            "detail": "File revision conflict.",
            "extra": "Read the latest revision.",
        },
        request=context.request,
    )
    http_error = httpx.HTTPStatusError(
        "Conflict", request=context.request, response=response
    )
    call_next = (
        Mock(side_effect=http_error) if raised_by_next else Mock(return_value=response)
    )

    with pytest.raises(FlowerError) as caught:
        FlowerErrorHttpInterceptor().intercept(context, call_next)

    assert caught.value.code == ApiErrorCode.RUNTIME_FILE_REVISION_CONFLICT
    assert caught.value.message == "File revision conflict."
    assert caught.value.public_details == "Read the latest revision."
    assert isinstance(caught.value.__cause__, httpx.HTTPStatusError)
    assert caught.value.__cause__.response is response
    if raised_by_next:
        assert caught.value.__cause__ is http_error


@pytest.mark.parametrize("content", [b"Bad Gateway", b'{"detail": "Not found"}'])
def test_preserves_unstructured_http_error(
    context: ProtobufRequestContext, content: bytes
) -> None:
    """Reraise the original HTTP exception when its body is not a Flower error."""
    response = httpx.Response(502, content=content, request=context.request)
    error = httpx.HTTPStatusError(
        "Bad Gateway", request=context.request, response=response
    )

    with pytest.raises(httpx.HTTPStatusError) as caught:
        FlowerErrorHttpInterceptor().intercept(context, Mock(side_effect=error))

    assert caught.value is error


def test_preserves_transport_error(context: ProtobufRequestContext) -> None:
    """Leave failures without an HTTP response unchanged."""
    error = httpx.ConnectError("Connection refused", request=context.request)

    with pytest.raises(httpx.ConnectError) as caught:
        FlowerErrorHttpInterceptor().intercept(context, Mock(side_effect=error))

    assert caught.value is error
