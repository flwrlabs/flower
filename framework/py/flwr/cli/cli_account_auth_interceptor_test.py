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
"""Tests for the CLI account auth interceptor."""


from collections import namedtuple
from unittest.mock import Mock

import grpc

from flwr.proto.control_pb2 import StreamRunEventsRequest  # pylint: disable=E0611

from .cli_account_auth_interceptor import CliAccountAuthInterceptor

_ClientCallDetails = namedtuple(
    "_ClientCallDetails",
    ["method", "timeout", "metadata", "credentials", "wait_for_ready", "compression"],
)


def test_stream_run_events_request_sends_auth_metadata() -> None:
    """StreamRunEvents requests should include account auth metadata."""
    auth_plugin = Mock()
    auth_plugin.write_tokens_to_metadata.return_value = [
        ("authorization", "Bearer token")
    ]
    response = Mock()
    response.initial_metadata.return_value = ()
    interceptor = CliAccountAuthInterceptor(auth_plugin)
    details = _ClientCallDetails(
        method="/flwr.proto.Control/StreamRunEvents",
        timeout=None,
        metadata=(),
        credentials=None,
        wait_for_ready=None,
        compression=None,
    )
    captured: dict[str, object] = {}

    def continuation(
        client_call_details: grpc.ClientCallDetails,
        request: StreamRunEventsRequest,
    ) -> Mock:
        captured["metadata"] = client_call_details.metadata
        captured["request"] = request
        return response

    result = interceptor.intercept_unary_stream(
        continuation=continuation,
        client_call_details=details,
        request=StreamRunEventsRequest(run_id=123),
    )

    assert result is response
    assert captured["metadata"] == [("authorization", "Bearer token")]
    assert captured["request"] == StreamRunEventsRequest(run_id=123)
    auth_plugin.write_tokens_to_metadata.assert_called_once_with([])
