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
"""Tests for reusable protobuf-over-HTTP client infrastructure."""

from unittest.mock import patch

import pytest
import requests

from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    ClaimTaskRequest,
    ClaimTaskResponse,
)
from flwr.supercore.protobuf.constants import PROTOBUF_MEDIA_TYPE

from .client import ProtobufCall, ProtobufClient, ProtobufRequestContext

_PATH = "/v1/runtime/claim-task"
_METHOD = "/flwr.proto.Runtime/ClaimTask"
_REQUEST = ClaimTaskRequest(task_id=123)
_RESPONSE = ClaimTaskResponse(token="task-token")


def _response(status_code: int, content: bytes = b"") -> requests.Response:
    """Create a requests response with a readable body."""
    response = requests.Response()
    response.status_code = status_code
    response._content = content  # pylint: disable=protected-access
    return response


def _call(client: ProtobufClient) -> ClaimTaskResponse:
    """Call one representative unary protobuf operation."""
    return client._unary_unary(  # pylint: disable=protected-access
        path=_PATH,
        rpc_method=_METHOD,
        request=_REQUEST,
        response_type=ClaimTaskResponse,
    )


def test_unary_unary_sends_and_receives_protobuf() -> None:
    """Serialize a protobuf request and parse its protobuf response."""
    response = _response(200, _RESPONSE.SerializeToString())
    with patch(
        "flwr.supercore.protobuf.client.requests.Session.send",
        return_value=response,
    ) as send:
        result = _call(
            ProtobufClient(
                "https://api.example/",
                verify="ca.pem",
                timeout=10.0,
            )
        )

    assert result == _RESPONSE
    prepared_request = send.call_args.args[0]
    assert prepared_request.method == "POST"
    assert prepared_request.url == f"https://api.example{_PATH}"
    assert prepared_request.body == _REQUEST.SerializeToString(deterministic=True)
    assert prepared_request.headers["content-type"] == PROTOBUF_MEDIA_TYPE
    assert prepared_request.headers["accept"] == PROTOBUF_MEDIA_TYPE
    assert send.call_args.kwargs == {"verify": "ca.pem", "timeout": 10.0}


def test_unary_unary_normalizes_path() -> None:
    """Accept an operation path without a leading slash."""
    response = _response(200, _RESPONSE.SerializeToString())
    with patch(
        "flwr.supercore.protobuf.client.requests.Session.send",
        return_value=response,
    ) as send:
        client = ProtobufClient("http://api.example")
        # pylint: disable-next=protected-access
        client._unary_unary(
            path=_PATH.removeprefix("/"),
            rpc_method=_METHOD,
            request=_REQUEST,
            response_type=ClaimTaskResponse,
        )

    assert send.call_args.args[0].url == f"http://api.example{_PATH}"


def test_unary_unary_raises_for_http_error() -> None:
    """Preserve requests' standard HTTP status error behavior."""
    response = _response(500)
    response.request = requests.Request("POST", "http://api.example").prepare()
    with (
        patch(
            "flwr.supercore.protobuf.client.requests.Session.send",
            return_value=response,
        ),
        pytest.raises(requests.HTTPError),
    ):
        _call(ProtobufClient("http://api.example"))


def test_unary_unary_rejects_invalid_protobuf_response() -> None:
    """Reject response bodies that are not valid protobuf messages."""
    with (
        patch(
            "flwr.supercore.protobuf.client.requests.Session.send",
            return_value=_response(200, b"not-protobuf"),
        ),
        pytest.raises(ValueError, match="Invalid protobuf response payload"),
    ):
        _call(ProtobufClient("http://api.example"))


class _RecordingInterceptor:
    def __init__(self, name: str, events: list[str]) -> None:
        self._name = name
        self._events = events

    def intercept(
        self,
        context: ProtobufRequestContext,
        call_next: ProtobufCall,
    ) -> requests.Response:
        """Record execution around the next interceptor and HTTP transport."""
        self._events.append(f"{self._name} before")
        assert context.rpc_method == _METHOD
        assert context.message == _REQUEST
        context.request.headers[f"x-{self._name}"] = "present"
        response = call_next(context)
        self._events.append(f"{self._name} after {response.status_code}")
        return response


def test_interceptors_wrap_request_in_configuration_order() -> None:
    """Expose request and response processing with middleware ordering."""
    events: list[str] = []

    def send(request: requests.PreparedRequest, **_kwargs: object) -> requests.Response:
        events.append("send")
        assert request.headers["x-A"] == "present"
        assert request.headers["x-B"] == "present"
        return _response(200, _RESPONSE.SerializeToString())

    client = ProtobufClient(
        "http://api.example",
        interceptors=[
            _RecordingInterceptor("A", events),
            _RecordingInterceptor("B", events),
        ],
    )
    with patch(
        "flwr.supercore.protobuf.client.requests.Session.send", side_effect=send
    ):
        _call(client)

    assert events == [
        "A before",
        "B before",
        "send",
        "B after 200",
        "A after 200",
    ]


def test_close_closes_session() -> None:
    """Close the underlying HTTP session."""
    with patch("flwr.supercore.protobuf.client.requests.Session") as session_class:
        client = ProtobufClient("http://api.example")
        client.close()

    session_class.return_value.close.assert_called_once_with()


def test_context_manager_returns_client_and_closes_session() -> None:
    """Manage the HTTP session with a context manager."""
    with patch("flwr.supercore.protobuf.client.requests.Session") as session_class:
        client = ProtobufClient("http://api.example")
        with client as entered_client:
            assert entered_client is client

    session_class.return_value.close.assert_called_once_with()
