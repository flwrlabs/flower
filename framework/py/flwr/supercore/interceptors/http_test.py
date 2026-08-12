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
"""Tests for reusable protobuf-over-HTTP client interceptors."""

from datetime import UTC, datetime
from logging import WARN
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
    FLWR_COMPONENT_NAME_METADATA_KEY,
    FLWR_PACKAGE_NAME_METADATA_KEY,
    FLWR_PACKAGE_VERSION_METADATA_KEY,
    SUPEREXEC_AUTH_BODY_SHA256_HEADER,
    SUPEREXEC_AUTH_NONCE_HEADER,
    SUPEREXEC_AUTH_SIGNATURE_HEADER,
    SUPEREXEC_AUTH_TIMESTAMP_HEADER,
    VERSION_INCOMPATIBILITY_MESSAGE_METADATA_KEY,
)
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.exit import ExitCode
from flwr.supercore.protobuf.client import ProtobufRequestContext

from .http import RuntimeVersionHttpInterceptor, SuperExecAuthHttpInterceptor
from .superexec_auth_interceptor import RUNTIME_SUPEREXEC_METHODS

_METHOD = "/flwr.proto.Runtime/PullPendingTasks"
_TIMESTAMP = 1000
_NONCE = "nonce"


def _context() -> ProtobufRequestContext:
    """Create a representative protobuf HTTP request context."""
    message = PullPendingTasksRequest()
    return ProtobufRequestContext(
        rpc_method=_METHOD,
        message=message,
        request=requests.Request("POST", "http://runtime.example").prepare(),
    )


def _response(
    status_code: int = 200,
    content: bytes = b"",
    headers: dict[str, str] | None = None,
) -> requests.Response:
    """Create a requests response with a readable body."""
    response = requests.Response()
    response.status_code = status_code
    response._content = content  # pylint: disable=protected-access
    response.headers.update(headers or {})
    return response


@patch(
    "flwr.supercore.auth.superexec.now",
    return_value=datetime.fromtimestamp(_TIMESTAMP, UTC),
)
@patch("flwr.supercore.auth.superexec.secrets.token_hex", return_value=_NONCE)
def test_superexec_auth_http_interceptor_signs_protected_request(
    _token_hex: Mock,
    _now: Mock,
) -> None:
    """Attach verifiable SuperExec authentication headers."""
    master_secret = b"master-secret"
    context = _context()
    response = _response()
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
        method=_METHOD,
        timestamp=_TIMESTAMP,
        nonce=_NONCE,
        body_sha256=body_sha256,
    )
    call_next.assert_called_once_with(context)


def test_superexec_auth_http_interceptor_skips_unprotected_request() -> None:
    """Leave requests outside the configured method set unsigned."""
    context = _context()
    context = ProtobufRequestContext(
        rpc_method="/flwr.proto.Runtime/Other",
        message=context.message,
        request=context.request,
    )
    interceptor = SuperExecAuthHttpInterceptor(
        master_secret=b"master-secret",
        protected_methods=RUNTIME_SUPEREXEC_METHODS,
    )

    interceptor.intercept(context, Mock(return_value=_response()))

    assert SUPEREXEC_AUTH_SIGNATURE_HEADER not in context.request.headers


def test_superexec_auth_http_interceptor_rejects_duplicate_headers() -> None:
    """Reject ambiguous headers instead of silently overwriting them."""
    context = _context()
    context.request.headers[SUPEREXEC_AUTH_TIMESTAMP_HEADER] = "existing"
    interceptor = SuperExecAuthHttpInterceptor(
        master_secret=b"master-secret",
        protected_methods=RUNTIME_SUPEREXEC_METHODS,
    )

    with pytest.raises(RuntimeError, match=SUPEREXEC_AUTH_TIMESTAMP_HEADER):
        interceptor.intercept(context, Mock(return_value=_response()))


def test_runtime_version_http_interceptor_adds_headers() -> None:
    """Attach local runtime-version information to an HTTP request."""
    context = _context()
    interceptor = RuntimeVersionHttpInterceptor(component_name="SuperExec")

    interceptor.intercept(context, Mock(return_value=_response()))

    assert FLWR_PACKAGE_NAME_METADATA_KEY in context.request.headers
    assert FLWR_PACKAGE_VERSION_METADATA_KEY in context.request.headers
    assert context.request.headers[FLWR_COMPONENT_NAME_METADATA_KEY] == "SuperExec"


def test_runtime_version_http_interceptor_logs_warning() -> None:
    """Log compatibility warnings returned in HTTP headers."""
    interceptor = RuntimeVersionHttpInterceptor(component_name="SuperExec")
    response = _response(
        headers={VERSION_INCOMPATIBILITY_MESSAGE_METADATA_KEY: "version warning"}
    )

    with patch("flwr.supercore.interceptors.http.log") as log_mock:
        interceptor.intercept(_context(), Mock(return_value=response))

    log_mock.assert_called_once_with(WARN, "version warning")


def test_runtime_version_http_interceptor_exits_on_incompatibility() -> None:
    """Use the established exit path for HTTP version incompatibility errors."""
    error = FlowerError(
        ApiErrorCode.RUNTIME_VERSION_INCOMPATIBLE,
        "internal",
        public_details="version details",
    ).to_json("Runtime version compatibility check failed.")
    interceptor = RuntimeVersionHttpInterceptor(component_name="SuperExec")

    with patch("flwr.supercore.interceptors.http.flwr_exit") as exit_mock:
        interceptor.intercept(
            _context(), Mock(return_value=_response(400, error.encode()))
        )

    exit_mock.assert_called_once_with(
        ExitCode.RUNTIME_VERSION_INCOMPATIBLE,
        "Runtime version compatibility check failed.\nversion details",
    )
