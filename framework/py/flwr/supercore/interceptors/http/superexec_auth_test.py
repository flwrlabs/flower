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
"""Tests for the SuperExec protobuf-over-HTTP client interceptor."""

from datetime import UTC, datetime
from unittest.mock import Mock, patch

import pytest
import requests

from flwr.proto.runtime_pb2 import PullPendingTasksRequest  # pylint: disable=E0611
from flwr.supercore.auth import (
    compute_request_body_sha256,
    compute_superexec_signature,
    derive_auth_secret,
)
from flwr.supercore.constant import (
    SUPEREXEC_AUTH_BODY_SHA256_HEADER,
    SUPEREXEC_AUTH_NONCE_HEADER,
    SUPEREXEC_AUTH_SIGNATURE_HEADER,
    SUPEREXEC_AUTH_TIMESTAMP_HEADER,
)
from flwr.supercore.interceptors.superexec_auth_interceptor import (
    RUNTIME_SUPEREXEC_METHODS,
)
from flwr.supercore.protobuf.client import ProtobufRequestContext

from .superexec_auth import SuperExecAuthHttpInterceptor

_RPC_METHOD = "/flwr.proto.Runtime/PullPendingTasks"
_TIMESTAMP = 1000
_NONCE = "nonce"


def _context(rpc_method: str = _RPC_METHOD) -> ProtobufRequestContext:
    """Create a representative protobuf HTTP request context."""
    return ProtobufRequestContext(
        rpc_method=rpc_method,
        message=PullPendingTasksRequest(),
        request=requests.Request("POST", "http://runtime.example").prepare(),
    )


@patch(
    "flwr.supercore.auth.superexec.now",
    return_value=datetime.fromtimestamp(_TIMESTAMP, UTC),
)
@patch("flwr.supercore.auth.superexec.secrets.token_hex", return_value=_NONCE)
def test_signs_protected_request(_token_hex: Mock, _now: Mock) -> None:
    """Attach verifiable SuperExec authentication headers."""
    master_secret = b"master-secret"
    context = _context()
    response = requests.Response()
    call_next = Mock(return_value=response)
    interceptor = SuperExecAuthHttpInterceptor(
        master_secret=master_secret,
        protected_methods=RUNTIME_SUPEREXEC_METHODS,
    )

    assert interceptor.intercept(context, call_next) is response

    body_sha256 = compute_request_body_sha256(context.message)
    assert context.request.headers[SUPEREXEC_AUTH_TIMESTAMP_HEADER] == str(_TIMESTAMP)
    assert context.request.headers[SUPEREXEC_AUTH_NONCE_HEADER] == _NONCE
    assert context.request.headers[SUPEREXEC_AUTH_BODY_SHA256_HEADER] == body_sha256
    assert context.request.headers[
        SUPEREXEC_AUTH_SIGNATURE_HEADER
    ] == compute_superexec_signature(
        auth_secret=derive_auth_secret(master_secret),
        method=_RPC_METHOD,
        timestamp=_TIMESTAMP,
        nonce=_NONCE,
        body_sha256=body_sha256,
    )
    call_next.assert_called_once_with(context)


def test_skips_unprotected_request() -> None:
    """Leave requests outside the configured method set unsigned."""
    context = _context("/flwr.proto.Runtime/Other")
    interceptor = SuperExecAuthHttpInterceptor(
        master_secret=b"master-secret",
        protected_methods=RUNTIME_SUPEREXEC_METHODS,
    )

    interceptor.intercept(context, Mock(return_value=requests.Response()))

    assert SUPEREXEC_AUTH_SIGNATURE_HEADER not in context.request.headers


def test_rejects_duplicate_headers() -> None:
    """Reject ambiguous headers instead of silently overwriting them."""
    context = _context()
    context.request.headers[SUPEREXEC_AUTH_TIMESTAMP_HEADER] = "existing"
    interceptor = SuperExecAuthHttpInterceptor(
        master_secret=b"master-secret",
        protected_methods=RUNTIME_SUPEREXEC_METHODS,
    )

    with pytest.raises(RuntimeError, match=SUPEREXEC_AUTH_TIMESTAMP_HEADER):
        interceptor.intercept(context, Mock(return_value=requests.Response()))
