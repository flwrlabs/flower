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
"""Tests for the Runtime HTTP client."""

import json
from unittest.mock import Mock, patch

import httpx
import pytest

from flwr.supercore.protobuf.client import ProtobufClient
from flwr.supercore.runtime import RuntimeHttpClient

_UNARY_UNARY_PATHS = (
    "pull-pending-tasks",
    "claim-task",
    "send-task-heartbeat",
    "pull-task-input",
    "push-task-output",
    "push-object",
    "pull-object",
    "confirm-message-received",
    "push-messages",
    "pull-messages",
    "push-logs",
    "get-nodes",
    "create-task",
    "start-automation",
    "push-task-message",
    "push-task-events",
    "pull-task-message",
    "record-task-usage",
    "get-connector",
)
_RESPONSE_NAME_OVERRIDES = {
    "push-messages": "PushAppMessagesResponse",
    "pull-messages": "PullAppMessagesResponse",
}


@pytest.mark.parametrize(
    "endpoint",
    _UNARY_UNARY_PATHS,
)
def test_runtime_method(endpoint: str) -> None:
    """Call one shared Runtime HTTP endpoint."""
    method_name = endpoint.title().replace("-", "")
    request = Mock()
    response = Mock()
    client = RuntimeHttpClient("http://runtime.example")

    with patch.object(ProtobufClient, "_unary_unary", return_value=response) as call:
        result = getattr(client, method_name)(request)

    assert result is response
    call.assert_called_once()
    assert call.call_args.kwargs["path"] == f"/v1/runtime/{endpoint}"
    assert call.call_args.kwargs["rpc_method"] == f"/flwr.proto.Runtime/{method_name}"
    assert call.call_args.kwargs["request"] is request
    expected_response_name = _RESPONSE_NAME_OVERRIDES.get(
        endpoint, f"{method_name}Response"
    )
    assert call.call_args.kwargs["response_type"].__name__ == expected_response_name


@pytest.mark.parametrize("stream", [False, True])
def test_create_response(stream: bool) -> None:
    """Return the final response for JSON and streaming requests."""
    request_payload = {"model": "model", "input": "hello", "stream": stream}
    response_payload = {
        "object": "response",
        "id": "resp-1",
        "status": "completed",
        "output": [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://runtime.example/v1/runtime/responses"
        assert request.headers["authorization"] == "Bearer task-token"
        assert json.loads(request.content) == request_payload
        if stream:
            completed_event = {
                "type": "response.completed",
                "response": response_payload,
            }
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=(
                    'event: response.created\ndata: {"type":"response.created"}\n\n'
                    "event: response.completed\n"
                    f"data: {json.dumps(completed_event)}\n\n"
                ),
            )
        return httpx.Response(200, json=response_payload)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    with patch("flwr.supercore.protobuf.client.httpx.Client", return_value=http_client):
        client = RuntimeHttpClient("http://runtime.example")

    try:
        result = client.create_response(
            request_payload, token="task-token", timeout=300.0
        )
    finally:
        client.close()

    assert result == response_payload
