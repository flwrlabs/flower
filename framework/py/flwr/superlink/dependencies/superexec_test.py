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
"""Tests for the FastAPI SuperExec authentication dependency."""

from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI, Request

from flwr.proto.runtime_pb2 import ClaimTaskRequest  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkState
from flwr.supercore.constant import (
    SUPEREXEC_AUTH_BODY_SHA256_HEADER,
    SUPEREXEC_AUTH_NONCE_HEADER,
    SUPEREXEC_AUTH_SIGNATURE_HEADER,
    SUPEREXEC_AUTH_TIMESTAMP_HEADER,
)
from flwr.supercore.error import ApiErrorCode, FlowerError

from .superexec import SuperExecAuthDependency

_METHOD = "/flwr.proto.Runtime/ClaimTask"
_HEADERS = {
    SUPEREXEC_AUTH_TIMESTAMP_HEADER: "1000",
    SUPEREXEC_AUTH_NONCE_HEADER: "nonce",
    SUPEREXEC_AUTH_BODY_SHA256_HEADER: "body-hash",
    SUPEREXEC_AUTH_SIGNATURE_HEADER: "signature",
}


def _make_request(
    master_secret: bytes | None,
    headers: list[tuple[str, str]] | None = None,
) -> tuple[Request, ClaimTaskRequest]:
    """Return a request carrying parsed protobuf and SuperExec headers."""
    headers = list(_HEADERS.items()) if headers is None else headers
    app = FastAPI()
    app.state.superexec_auth_secret = master_secret
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/runtime/claim-task",
            "headers": [(key.encode(), value.encode()) for key, value in headers],
            "query_string": b"",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "scheme": "http",
            "app": app,
        }
    )
    protobuf_request = ClaimTaskRequest(task_id=11)
    request.state.protobuf_request = protobuf_request
    return request, protobuf_request


def test_superexec_auth_is_optional() -> None:
    """Skip verification when no SuperExec secret is configured."""
    request, _ = _make_request(None)

    with patch(
        "flwr.superlink.dependencies.superexec.verify_superexec_request"
    ) as verify:
        SuperExecAuthDependency(_METHOD)(request, Mock(spec=LinkState))

    verify.assert_not_called()


def test_superexec_auth_verifies_request() -> None:
    """Pass HTTP authentication fields to the shared verifier."""
    request, protobuf_request = _make_request(b"master-secret")
    state = Mock(spec=LinkState)

    with (
        patch(
            "flwr.superlink.dependencies.superexec.derive_auth_secret",
            return_value=b"derived-secret",
        ),
        patch(
            "flwr.superlink.dependencies.superexec.verify_superexec_request",
            return_value=True,
        ) as verify,
    ):
        SuperExecAuthDependency(_METHOD)(request, state)

    verify.assert_called_once_with(
        request=protobuf_request,
        method=_METHOD,
        auth_secret=b"derived-secret",
        timestamp_raw="1000",
        nonce="nonce",
        body_sha256_header="body-hash",
        signature="signature",
        nonce_store=state,
    )


def test_superexec_auth_rejects_failed_verification() -> None:
    """Raise the Runtime authentication error when verification fails."""
    request, _ = _make_request(b"master-secret")

    with (
        patch(
            "flwr.superlink.dependencies.superexec.verify_superexec_request",
            return_value=False,
        ),
        pytest.raises(FlowerError) as exc_info,
    ):
        SuperExecAuthDependency(_METHOD)(request, Mock(spec=LinkState))

    assert exc_info.value.code == ApiErrorCode.RUNTIME_AUTHENTICATION_FAILED
    assert exc_info.value.message == (f"SuperExec authentication failed for {_METHOD}.")


@pytest.mark.parametrize(
    "timestamp_headers",
    [
        [],
        [(SUPEREXEC_AUTH_TIMESTAMP_HEADER, "")],
        [(SUPEREXEC_AUTH_TIMESTAMP_HEADER, "1000")] * 2,
    ],
)
def test_superexec_auth_rejects_invalid_header_count_or_value(
    timestamp_headers: list[tuple[str, str]],
) -> None:
    """Reject missing, empty, or duplicate SuperExec authentication headers."""
    other_headers = [
        (name, value)
        for name, value in _HEADERS.items()
        if name != SUPEREXEC_AUTH_TIMESTAMP_HEADER
    ]
    request, _ = _make_request(
        b"master-secret", headers=timestamp_headers + other_headers
    )

    with (
        patch(
            "flwr.superlink.dependencies.superexec.verify_superexec_request"
        ) as verify,
        pytest.raises(FlowerError) as exc_info,
    ):
        SuperExecAuthDependency(_METHOD)(request, Mock(spec=LinkState))

    assert exc_info.value.code == ApiErrorCode.RUNTIME_AUTHENTICATION_FAILED
    verify.assert_not_called()
