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

import ssl
from unittest.mock import Mock, patch

import httpx
import pytest

from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    ClaimTaskRequest,
    ClaimTaskResponse,
)
from flwr.supercore.protobuf.constants import PROTOBUF_MEDIA_TYPE

from .client import (
    ProtobufCall,
    ProtobufClient,
    ProtobufRequestContext,
    create_protobuf_client,
)

_PATH = "/v1/runtime/claim-task"
_METHOD = "/flwr.proto.Runtime/ClaimTask"
_REQUEST = ClaimTaskRequest(task_id=123)
_RESPONSE = ClaimTaskResponse(token="task-token")


def _response(status_code: int, content: bytes = b"") -> httpx.Response:
    """Create an HTTP response."""
    return httpx.Response(
        status_code,
        content=content,
        request=httpx.Request("POST", "http://api.example"),
    )


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
        "flwr.supercore.protobuf.client.httpx.Client.send",
        return_value=response,
    ) as send:
        result = _call(
            ProtobufClient(
                "https://api.example/",
                verify=False,
                timeout=10.0,
            )
        )

    assert result == _RESPONSE
    http_request = send.call_args.args[0]
    assert http_request.method == "POST"
    assert str(http_request.url) == f"https://api.example{_PATH}"
    assert http_request.content == _REQUEST.SerializeToString(deterministic=True)
    assert http_request.headers["content-type"] == PROTOBUF_MEDIA_TYPE
    assert http_request.headers["accept"] == PROTOBUF_MEDIA_TYPE
    assert send.call_args.kwargs == {}


def test_configures_http_client() -> None:
    """Pass TLS verification and timeout settings to the HTTP client."""
    with patch("flwr.supercore.protobuf.client.httpx.Client") as client_class:
        ProtobufClient(
            "https://api.example",
            verify="ca.pem",
            timeout=10.0,
        )

    client_class.assert_called_once_with(
        verify="ca.pem",
        timeout=10.0,
        follow_redirects=True,
    )


def test_unary_unary_normalizes_path() -> None:
    """Accept an operation path without a leading slash."""
    response = _response(200, _RESPONSE.SerializeToString())
    with patch(
        "flwr.supercore.protobuf.client.httpx.Client.send",
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

    assert str(send.call_args.args[0].url) == f"http://api.example{_PATH}"


def test_unary_unary_raises_for_http_error() -> None:
    """Preserve httpx's standard HTTP status error behavior."""
    response = httpx.Response(500, request=httpx.Request("POST", "http://api.example"))
    with (
        patch(
            "flwr.supercore.protobuf.client.httpx.Client.send",
            return_value=response,
        ),
        pytest.raises(httpx.HTTPStatusError),
    ):
        _call(ProtobufClient("http://api.example"))


def test_unary_unary_rejects_invalid_protobuf_response() -> None:
    """Reject response bodies that are not valid protobuf messages."""
    with (
        patch(
            "flwr.supercore.protobuf.client.httpx.Client.send",
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
    ) -> httpx.Response:
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

    def send(request: httpx.Request, **_kwargs: object) -> httpx.Response:
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
    with patch("flwr.supercore.protobuf.client.httpx.Client.send", side_effect=send):
        _call(client)

    assert events == [
        "A before",
        "B before",
        "send",
        "B after 200",
        "A after 200",
    ]


def test_close_closes_client() -> None:
    """Close the underlying HTTP client."""
    with patch("flwr.supercore.protobuf.client.httpx.Client") as client_class:
        client = ProtobufClient("http://api.example")
        client.close()

    client_class.return_value.close.assert_called_once_with()


def test_context_manager_returns_client_and_closes_client() -> None:
    """Manage the HTTP client with a context manager."""
    with patch("flwr.supercore.protobuf.client.httpx.Client") as client_class:
        client = ProtobufClient("http://api.example")
        with client as entered_client:
            assert entered_client is client

    client_class.return_value.close.assert_called_once_with()


def test_create_protobuf_client_uses_plain_http_when_insecure() -> None:
    """Create an unverified HTTP client in insecure mode."""
    client_class = Mock()

    create_protobuf_client(
        client_class=client_class,
        server_address="127.0.0.1:8000",
        insecure=True,
        root_certificates=None,
        interceptors=[],
    )

    client_class.assert_called_once_with(
        "http://127.0.0.1:8000", interceptors=[], verify=False
    )


@pytest.mark.parametrize("root_certificates", [b"certificate", "ca.pem"])
def test_create_protobuf_client_rejects_certificates_when_insecure(
    root_certificates: bytes | str,
) -> None:
    """Reject root certificates for a plaintext connection."""
    with pytest.raises(ValueError, match="root_certificates.*insecure"):
        create_protobuf_client(
            client_class=Mock(),
            server_address="127.0.0.1:8000",
            insecure=True,
            root_certificates=root_certificates,
            interceptors=[],
        )


def test_create_protobuf_client_uses_certificate_path() -> None:
    """Pass a CA certificate path to the HTTP client."""
    client_class = Mock()

    create_protobuf_client(
        client_class=client_class,
        server_address="api.example:443",
        insecure=False,
        root_certificates="ca.pem",
        interceptors=[],
    )

    client_class.assert_called_once_with(
        "https://api.example:443", interceptors=[], verify="ca.pem"
    )


def test_create_protobuf_client_loads_certificate_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load in-memory CA certificates into an SSL context."""
    client_class = Mock()
    context = Mock(spec=ssl.SSLContext)
    ssl_context = Mock(return_value=context)
    monkeypatch.setattr(ssl, "SSLContext", ssl_context)

    create_protobuf_client(
        client_class=client_class,
        server_address="api.example:443",
        insecure=False,
        root_certificates=b"certificate",
        interceptors=[],
    )

    ssl_context.assert_called_once_with(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations.assert_called_once_with(cadata="certificate")
    client_class.assert_called_once_with(
        "https://api.example:443", interceptors=[], verify=context
    )
