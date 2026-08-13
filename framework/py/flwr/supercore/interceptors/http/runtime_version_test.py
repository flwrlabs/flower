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
"""Tests for the runtime-version protobuf-over-HTTP client interceptor."""

from logging import WARN
from unittest.mock import Mock, patch

import requests

from flwr.proto.runtime_pb2 import PullPendingTasksRequest  # pylint: disable=E0611
from flwr.supercore.constant import (
    FLWR_COMPONENT_NAME_METADATA_KEY,
    FLWR_PACKAGE_NAME_METADATA_KEY,
    FLWR_PACKAGE_VERSION_METADATA_KEY,
    VERSION_INCOMPATIBILITY_MESSAGE_METADATA_KEY,
)
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.exit import ExitCode
from flwr.supercore.protobuf.client import ProtobufRequestContext

from .runtime_version import RuntimeVersionHttpInterceptor


def _context() -> ProtobufRequestContext:
    """Create a representative protobuf HTTP request context."""
    return ProtobufRequestContext(
        rpc_method="/flwr.proto.Runtime/PullPendingTasks",
        message=PullPendingTasksRequest(),
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


def test_adds_headers() -> None:
    """Attach local runtime-version information to an HTTP request."""
    context = _context()
    interceptor = RuntimeVersionHttpInterceptor(component_name="SuperExec")

    interceptor.intercept(context, Mock(return_value=_response()))

    assert FLWR_PACKAGE_NAME_METADATA_KEY in context.request.headers
    assert FLWR_PACKAGE_VERSION_METADATA_KEY in context.request.headers
    assert context.request.headers[FLWR_COMPONENT_NAME_METADATA_KEY] == "SuperExec"


def test_logs_warning() -> None:
    """Log compatibility warnings returned in HTTP headers."""
    interceptor = RuntimeVersionHttpInterceptor(component_name="SuperExec")
    response = _response(
        headers={VERSION_INCOMPATIBILITY_MESSAGE_METADATA_KEY: "version warning"}
    )

    with patch("flwr.supercore.interceptors.http.runtime_version.log") as log_mock:
        interceptor.intercept(_context(), Mock(return_value=response))

    log_mock.assert_called_once_with(WARN, "version warning")


def test_exits_on_incompatibility() -> None:
    """Use the established exit path for HTTP version incompatibility errors."""
    error = FlowerError(
        ApiErrorCode.RUNTIME_VERSION_INCOMPATIBLE,
        "internal",
        public_details="version details",
    ).to_json("Runtime version compatibility check failed.")
    interceptor = RuntimeVersionHttpInterceptor(component_name="SuperExec")

    with patch(
        "flwr.supercore.interceptors.http.runtime_version.flwr_exit"
    ) as exit_mock:
        interceptor.intercept(
            _context(), Mock(return_value=_response(400, error.encode()))
        )

    exit_mock.assert_called_once_with(
        ExitCode.RUNTIME_VERSION_INCOMPATIBLE,
        "Runtime version compatibility check failed.\nversion details",
    )
