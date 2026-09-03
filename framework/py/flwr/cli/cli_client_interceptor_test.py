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
# ===============================================================================
"""Tests for the Flower CLI client interceptors."""

from unittest.mock import Mock

import httpx

from flwr.proto.control_pb2 import ListRunsRequest  # pylint: disable=E0611
from flwr.supercore.constant import FLWR_CLIENT_METADATA_KEY
from flwr.supercore.protobuf.client import ProtobufRequestContext

from .cli_client_interceptor import CliClientHttpInterceptor


def test_http_interceptor_sets_client_header() -> None:
    """Set the authoritative CLI identifier and preserve other headers."""
    context = ProtobufRequestContext(
        rpc_method="/flwr.proto.Control/ListRuns",
        message=ListRunsRequest(),
        request=httpx.Request(
            "POST",
            "https://control.example",
            headers={FLWR_CLIENT_METADATA_KEY: "web_ui", "x-test": "value"},
        ),
    )
    response = httpx.Response(200)
    call_next = Mock(return_value=response)

    result = CliClientHttpInterceptor().intercept(context, call_next)

    assert result is response
    assert context.request.headers[FLWR_CLIENT_METADATA_KEY] == "cli"
    assert context.request.headers["x-test"] == "value"
    call_next.assert_called_once_with(context)
