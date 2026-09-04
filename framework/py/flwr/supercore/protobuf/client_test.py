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

import gzip
import random
import ssl
from collections.abc import Generator, Iterator
from unittest.mock import Mock, patch

import httpx
import pytest

from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    ClaimTaskRequest,
    ClaimTaskResponse,
)
from flwr.supercore.constant import MAX_PROTOBUF_STREAM_MESSAGE_LENGTH
from flwr.supercore.interceptors import (
    RuntimeTokenHttpInterceptor,
    RuntimeVersionHttpInterceptor,
)
from flwr.supercore.protobuf.constants import (
    FRAME_HEADER_SIZE,
    PROTOBUF_MEDIA_TYPE,
    PROTOBUF_STREAM_MEDIA_TYPE,
)
from flwr.supercore.protobuf.framing import frame_message
from flwr.supercore.retry import make_simple_http_retry_invoker

from .client import ProtobufCall, ProtobufClient, ProtobufRequestContext

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


def _stream_call(
    client: ProtobufClient, *, read_timeout: float | None = None
) -> Generator[ClaimTaskResponse, None, None]:
    """Call one representative streaming protobuf operation."""
    return client._unary_stream(  # pylint: disable=protected-access
        path=_PATH,
        rpc_method=_METHOD,
        request=_REQUEST,
        response_type=ClaimTaskResponse,
        read_timeout=read_timeout,
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


def test_unary_unary_uses_retry_invoker() -> None:
    """Retry a transient HTTP response before parsing the protobuf response."""
    retry_invoker = make_simple_http_retry_invoker()
    retry_invoker.max_tries = 2
    retry_invoker.jitter = None
    retry_invoker.wait_function = lambda _: None

    with patch(
        "flwr.supercore.protobuf.client.httpx.Client.send",
        side_effect=[_response(503), _response(200, _RESPONSE.SerializeToString())],
    ) as send:
        result = _call(
            ProtobufClient(
                "http://api.example",
                retry_invoker=retry_invoker,
            )
        )

    assert result == _RESPONSE
    assert send.call_count == 2


def test_retry_rebuilds_request_before_applying_interceptors() -> None:
    """Apply HTTP interceptors to a fresh request on every retry attempt."""
    retry_invoker = make_simple_http_retry_invoker()
    retry_invoker.max_tries = 2
    retry_invoker.jitter = None
    retry_invoker.wait_function = lambda _: None
    client = ProtobufClient(
        "http://api.example",
        interceptors=[RuntimeTokenHttpInterceptor("task-token")],
        retry_invoker=retry_invoker,
    )

    with patch(
        "flwr.supercore.protobuf.client.httpx.Client.send",
        side_effect=[_response(503), _response(200, _RESPONSE.SerializeToString())],
    ) as send:
        result = _call(client)

    assert result == _RESPONSE
    assert send.call_count == 2
    for call in send.call_args_list:
        assert call.args[0].headers["flwr-task-token"] == "task-token"


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


class _ChunkedByteStream(httpx.SyncByteStream):
    """Yield an HTTP response body using predefined chunk boundaries."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __iter__(self) -> Iterator[bytes]:
        yield from self._chunks


def _stream_response(status_code: int, chunks: list[bytes]) -> httpx.Response:
    """Create a streaming HTTP response."""
    return httpx.Response(
        status_code,
        stream=_ChunkedByteStream(chunks),
        request=httpx.Request("POST", "http://api.example"),
    )


def test_unary_stream_sends_and_receives_framed_protobuf() -> None:
    """Decode messages split across arbitrary HTTP response chunks."""
    first = ClaimTaskResponse(token="first")
    second = ClaimTaskResponse(token="second")
    content = frame_message(first) + frame_message(second)
    response = _stream_response(
        200,
        [content[:2], content[2:7], content[7:-1], content[-1:]],
    )

    with patch(
        "flwr.supercore.protobuf.client.httpx.Client.send",
        return_value=response,
    ) as send:
        result = list(
            _stream_call(ProtobufClient("https://api.example/", timeout=10.0))
        )

    assert result == [first, second]
    http_request = send.call_args.args[0]
    assert http_request.method == "POST"
    assert str(http_request.url) == f"https://api.example{_PATH}"
    assert http_request.content == _REQUEST.SerializeToString(deterministic=True)
    assert http_request.headers["content-type"] == PROTOBUF_MEDIA_TYPE
    assert http_request.headers["accept"] == PROTOBUF_STREAM_MEDIA_TYPE
    assert http_request.headers["accept-encoding"] == "identity"
    assert http_request.extensions["timeout"] == {
        "connect": 10.0,
        "read": None,
        "write": 10.0,
        "pool": 10.0,
    }
    assert send.call_args.kwargs == {"stream": True}
    assert response.is_closed


def test_unary_stream_sets_read_timeout() -> None:
    """Apply the optional timeout while reading a successful stream."""
    response = _stream_response(200, [])
    read_timeout = None

    def send(request: httpx.Request, **_kwargs: object) -> httpx.Response:
        nonlocal read_timeout
        read_timeout = request.extensions["timeout"]["read"]
        return response

    with patch(
        "flwr.supercore.protobuf.client.httpx.Client.send",
        side_effect=send,
    ):
        list(
            _stream_call(
                ProtobufClient("https://api.example/", timeout=10.0),
                read_timeout=5.0,
            )
        )

    assert read_timeout == 5.0


@pytest.mark.parametrize(
    "content",
    [
        b"\x00",
        len(b"partial").to_bytes(4, "big") + b"part",
    ],
)
def test_unary_stream_rejects_truncated_frame(content: bytes) -> None:
    """Reject a stream ending within a frame header or payload."""
    response = _stream_response(200, [content])
    with (
        patch(
            "flwr.supercore.protobuf.client.httpx.Client.send",
            return_value=response,
        ),
        pytest.raises(ValueError, match="Truncated protobuf stream frame"),
    ):
        list(_stream_call(ProtobufClient("http://api.example")))

    assert response.is_closed


def test_unary_stream_rejects_oversized_frame() -> None:
    """Reject an oversized frame before receiving its payload."""
    content = (MAX_PROTOBUF_STREAM_MESSAGE_LENGTH + 1).to_bytes(
        FRAME_HEADER_SIZE, "big"
    )
    response = _stream_response(200, [content])
    with (
        patch(
            "flwr.supercore.protobuf.client.httpx.Client.send",
            return_value=response,
        ),
        pytest.raises(ValueError, match="exceeds maximum"),
    ):
        list(_stream_call(ProtobufClient("http://api.example")))

    assert response.is_closed


def test_unary_stream_rejects_invalid_protobuf_payload() -> None:
    """Reject a complete frame containing an invalid protobuf message."""
    response = _stream_response(200, [len(b"invalid").to_bytes(4, "big") + b"invalid"])
    with (
        patch(
            "flwr.supercore.protobuf.client.httpx.Client.send",
            return_value=response,
        ),
        pytest.raises(ValueError, match="Invalid protobuf stream payload"),
    ):
        list(_stream_call(ProtobufClient("http://api.example")))

    assert response.is_closed


def test_unary_stream_closes_response_when_iteration_stops() -> None:
    """Close the HTTP response when a caller cancels stream iteration."""
    response = _stream_response(
        200,
        [
            frame_message(ClaimTaskResponse(token="first")),
            frame_message(ClaimTaskResponse(token="second")),
        ],
    )
    with patch(
        "flwr.supercore.protobuf.client.httpx.Client.send",
        return_value=response,
    ):
        messages = _stream_call(ProtobufClient("http://api.example"))
        assert next(messages).token == "first"
        messages.close()

    assert response.is_closed


def test_unary_stream_does_not_open_response_before_iteration() -> None:
    """Avoid acquiring a response for a stream that is never started."""
    response = _stream_response(200, [])
    with patch(
        "flwr.supercore.protobuf.client.httpx.Client.send",
        return_value=response,
    ) as send:
        messages = _stream_call(ProtobufClient("http://api.example"))
        messages.close()

    send.assert_not_called()
    assert not response.is_closed
    response.close()


def test_unary_stream_closes_response_for_http_error() -> None:
    """Close a streaming response before propagating an HTTP status error."""
    response = _stream_response(500, [])
    with (
        patch(
            "flwr.supercore.protobuf.client.httpx.Client.send",
            return_value=response,
        ),
        pytest.raises(httpx.HTTPStatusError),
    ):
        list(_stream_call(ProtobufClient("http://api.example")))

    assert response.is_closed


def test_unary_stream_bounds_error_response_body() -> None:
    """Bound the time and size used to read a streaming error response."""
    max_len = 64 * 1024
    compressed_content = gzip.compress(
        b"x" * (4 * max_len) + random.Random(0).randbytes(max_len)
    )
    assert len(compressed_content) > max_len
    response = httpx.Response(
        500,
        headers={"content-encoding": "gzip"},
        stream=_ChunkedByteStream([compressed_content]),
        request=httpx.Request("POST", "http://api.example"),
    )
    captured_responses: list[httpx.Response] = []
    interceptor = Mock()

    def capture_response(
        context: ProtobufRequestContext, call_next: ProtobufCall
    ) -> httpx.Response:
        buffered_response = call_next(context)
        captured_responses.append(buffered_response)
        return buffered_response

    interceptor.intercept.side_effect = capture_response
    with (
        patch(
            "flwr.supercore.protobuf.client.httpx.Client.send",
            return_value=response,
        ) as send,
        pytest.raises(httpx.HTTPStatusError),
    ):
        list(
            _stream_call(
                ProtobufClient(
                    "http://api.example",
                    interceptors=[interceptor],
                    timeout=10.0,
                )
            )
        )

    request = send.call_args.args[0]
    assert request.extensions["timeout"]["read"] == 10.0
    assert len(captured_responses[0].content) == max_len
    assert captured_responses[0].content == compressed_content[:max_len]
    assert "content-encoding" not in captured_responses[0].headers
    assert response.is_closed


def test_unary_stream_retries_only_before_returning_response() -> None:
    """Retry response establishment but not failures during stream iteration."""
    retry_invoker = make_simple_http_retry_invoker()
    retry_invoker.max_tries = 2
    retry_invoker.jitter = None
    retry_invoker.wait_function = lambda _: None
    unavailable = _stream_response(503, [])
    invalid_stream = _stream_response(
        200,
        [
            frame_message(ClaimTaskResponse(token="first")),
            len(b"invalid").to_bytes(4, "big") + b"invalid",
        ],
    )

    with patch(
        "flwr.supercore.protobuf.client.httpx.Client.send",
        side_effect=[unavailable, invalid_stream],
    ) as send:
        messages = _stream_call(
            ProtobufClient(
                "http://api.example",
                interceptors=[RuntimeVersionHttpInterceptor("test")],
                retry_invoker=retry_invoker,
            )
        )
        assert next(messages).token == "first"
        with pytest.raises(ValueError, match="Invalid protobuf stream payload"):
            next(messages)

    assert send.call_count == 2
    assert unavailable.is_closed
    assert invalid_stream.is_closed


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


def test_from_server_address_uses_plain_http_when_insecure() -> None:
    """Create an unverified HTTP client in insecure mode."""
    with patch("flwr.supercore.protobuf.client.httpx.Client") as http_client:
        client = ProtobufClient.from_server_address(
            server_address="127.0.0.1:8000",
            insecure=True,
            root_certificates=None,
            interceptors=[],
        )

    assert client._base_url == "http://127.0.0.1:8000"  # pylint: disable=W0212
    http_client.assert_called_once_with(
        verify=False, timeout=30.0, follow_redirects=True
    )


def test_from_server_address_uses_default_ca_bundle() -> None:
    """Use HTTPX's default trusted CA bundle for secure connections by default."""
    with patch("flwr.supercore.protobuf.client.httpx.Client") as http_client:
        client = ProtobufClient.from_server_address(
            server_address="api.example:443",
            insecure=False,
            root_certificates=None,
            interceptors=[],
        )

    assert client._base_url == "https://api.example:443"  # pylint: disable=W0212
    http_client.assert_called_once_with(
        verify=True, timeout=30.0, follow_redirects=True
    )


@pytest.mark.parametrize("root_certificates", [b"certificate", "ca.pem"])
def test_from_server_address_rejects_certificates_when_insecure(
    root_certificates: bytes | str,
) -> None:
    """Reject root certificates for a plaintext connection."""
    with pytest.raises(ValueError, match="root_certificates.*insecure"):
        ProtobufClient.from_server_address(
            server_address="127.0.0.1:8000",
            insecure=True,
            root_certificates=root_certificates,
            interceptors=[],
        )


def test_from_server_address_uses_certificate_path() -> None:
    """Pass a CA certificate path to the HTTP client."""
    with patch("flwr.supercore.protobuf.client.httpx.Client") as http_client:
        client = ProtobufClient.from_server_address(
            server_address="api.example:443",
            insecure=False,
            root_certificates="ca.pem",
            interceptors=[],
        )

    assert client._base_url == "https://api.example:443"  # pylint: disable=W0212
    http_client.assert_called_once_with(
        verify="ca.pem", timeout=30.0, follow_redirects=True
    )


def test_from_server_address_loads_certificate_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load in-memory CA certificates into an SSL context."""
    context = Mock(spec=ssl.SSLContext)
    ssl_context = Mock(return_value=context)
    monkeypatch.setattr(ssl, "SSLContext", ssl_context)

    with patch("flwr.supercore.protobuf.client.httpx.Client") as http_client:
        client = ProtobufClient.from_server_address(
            server_address="api.example:443",
            insecure=False,
            root_certificates=b"certificate",
            interceptors=[],
        )

    assert client._base_url == "https://api.example:443"  # pylint: disable=W0212
    ssl_context.assert_called_once_with(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations.assert_called_once_with(cadata="certificate")
    http_client.assert_called_once_with(
        verify=context, timeout=30.0, follow_redirects=True
    )
