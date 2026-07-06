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
"""Tests for the private web fetch provider."""

from __future__ import annotations

import socket
from collections.abc import Iterator
from dataclasses import dataclass, field
from unittest.mock import Mock

import pytest
import requests

from .web_fetch import (
    WEB_FETCH_ENDPOINT_ENV,
    WebFetchProviderError,
    invoke_web_fetch_provider,
)

_PROXY_ENDPOINT = "http://proxy/v1/web-fetch"


@dataclass
class _Response:  # pylint: disable=too-many-instance-attributes
    status_code: int = 200
    url: str = "https://example.com/final"
    headers: dict[str, str] = field(
        default_factory=lambda: {"Content-Type": "text/html; charset=utf-8"}
    )
    body: bytes = b"<html><body><main>Hello</main></body></html>"
    chunks: list[bytes] | None = None
    text: str = ""
    encoding: str | None = "utf-8"
    apparent_encoding: str | None = "utf-8"
    closed: bool = False

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        """Return the mocked response body in chunks."""
        del chunk_size
        if self.chunks is not None:
            yield from self.chunks
            return
        yield self.body

    def close(self) -> None:
        """Record that the response was closed."""
        self.closed = True


@dataclass
class _ProxyResponse:
    status_code: int = 200
    payload: object | None = None
    text: str = ""
    json_error: ValueError | None = None

    def json(self) -> object:
        """Return the mocked JSON response body."""
        if self.json_error is not None:
            raise self.json_error
        return self.payload


def _patch_dns(monkeypatch: pytest.MonkeyPatch, ip_address: str) -> None:
    """Patch DNS resolution to a deterministic address."""

    def getaddrinfo(
        host: str,
        port: int | None,
        *args: object,
        **kwargs: object,
    ) -> list[tuple[int, int, int, str, tuple[str, int]]]:
        """Return a test address for every hostname."""
        del host, args, kwargs
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip_address, port or 0))]

    monkeypatch.setattr(
        "flwr.supercore.task_process.connector.web_fetch.socket.getaddrinfo",
        getaddrinfo,
    )


@pytest.fixture(autouse=True)
def _resolve_hosts_to_public_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid real DNS lookups in provider tests."""
    monkeypatch.delenv(WEB_FETCH_ENDPOINT_ENV, raising=False)
    _patch_dns(monkeypatch, "93.184.216.34")


def _patch_get(
    monkeypatch: pytest.MonkeyPatch,
    response: _Response | list[_Response],
) -> Mock:
    if isinstance(response, list):
        get_mock = Mock(side_effect=response)
    else:
        get_mock = Mock(return_value=response)
    monkeypatch.setattr(
        "flwr.supercore.task_process.connector.web_fetch.requests.get",
        get_mock,
    )
    return get_mock


def test_invoke_web_fetch_provider_extracts_markdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requests should return trafilatura-extracted markdown."""
    trafilatura = pytest.importorskip("trafilatura")
    response = _Response()
    get_mock = _patch_get(monkeypatch, response)
    extract_mock = Mock(return_value="# Hello")
    monkeypatch.setattr(trafilatura, "extract", extract_mock)

    result = invoke_web_fetch_provider("https://example.com")

    assert result == {
        "object": "web_fetch.response",
        "status": "completed",
        "url": "https://example.com",
        "final_url": "https://example.com/final",
        "status_code": 200,
        "content_type": "text/html; charset=utf-8",
        "content": "# Hello",
    }
    get_mock.assert_called_once_with(
        "https://example.com",
        timeout=30.0,
        stream=True,
        allow_redirects=False,
    )
    extract_mock.assert_called_once()
    assert response.closed


def test_invoke_web_fetch_provider_calls_proxy_endpoint_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proxy mode should post the URL to the configured endpoint."""
    monkeypatch.setenv(WEB_FETCH_ENDPOINT_ENV, _PROXY_ENDPOINT)
    response = Mock(status_code=200)
    response.json.return_value = {"content": "Hello"}
    get_mock = Mock()
    post_mock = Mock(return_value=response)
    monkeypatch.setattr(
        "flwr.supercore.task_process.connector.web_fetch.requests.post",
        post_mock,
    )
    monkeypatch.setattr(
        "flwr.supercore.task_process.connector.web_fetch.requests.get",
        get_mock,
    )

    assert invoke_web_fetch_provider(" https://example.com:443 ") == {
        "content": "Hello"
    }

    post_mock.assert_called_once_with(
        _PROXY_ENDPOINT,
        json={"url": "https://example.com:443"},
        timeout=60.0,
    )
    get_mock.assert_not_called()


@pytest.mark.parametrize(
    ("post_result", "expected_code", "expected_status_code", "expected_message"),
    [
        (
            requests.RequestException("connection failed"),
            "fetch_failed",
            None,
            "proxy web fetch request failed",
        ),
        (
            _ProxyResponse(
                status_code=403,
                payload={"code": "blocked_url", "detail": "URL host is not allowed."},
            ),
            "http_error",
            403,
            "blocked_url",
        ),
        (
            _ProxyResponse(
                status_code=502,
                text="bad gateway",
                json_error=ValueError("not json"),
            ),
            "http_error",
            502,
            "bad gateway",
        ),
        (
            _ProxyResponse(json_error=ValueError("not json")),
            "invalid_response",
            None,
            "invalid JSON",
        ),
        (
            _ProxyResponse(payload=["not an object"]),
            "invalid_response",
            None,
            "invalid JSON",
        ),
        (
            _ProxyResponse(payload={}),
            "invalid_response",
            None,
            "must contain content",
        ),
        (
            _ProxyResponse(payload={"content": ["not a string"]}),
            "invalid_response",
            None,
            "must contain content",
        ),
    ],
)
def test_invoke_web_fetch_provider_reports_proxy_errors(
    monkeypatch: pytest.MonkeyPatch,
    post_result: requests.RequestException | _ProxyResponse,
    expected_code: str,
    expected_status_code: int | None,
    expected_message: str,
) -> None:
    """Proxy mode should report request, HTTP, and payload errors."""
    monkeypatch.setenv(WEB_FETCH_ENDPOINT_ENV, _PROXY_ENDPOINT)
    get_mock = Mock()
    if isinstance(post_result, requests.RequestException):
        post_mock = Mock(side_effect=post_result)
    else:
        post_mock = Mock(return_value=post_result)
    monkeypatch.setattr(
        "flwr.supercore.task_process.connector.web_fetch.requests.get",
        get_mock,
    )
    monkeypatch.setattr(
        "flwr.supercore.task_process.connector.web_fetch.requests.post",
        post_mock,
    )

    with pytest.raises(WebFetchProviderError, match=expected_message) as exc_info:
        invoke_web_fetch_provider("https://example.com")

    assert exc_info.value.code == expected_code
    assert exc_info.value.status_code == expected_status_code
    get_mock.assert_not_called()


@pytest.mark.parametrize(
    "blocked_url",
    [
        "https://user@example.com",
        "https://user:pass@example.com",
        "http://example.com:443",
        "https://example.com:80",
        "https://example.com:bad",
        "https://example.com:",
    ],
)
def test_invoke_web_fetch_provider_rejects_invalid_url_syntax(
    monkeypatch: pytest.MonkeyPatch,
    blocked_url: str,
) -> None:
    """Provider should reject unsafe URL syntax before any request."""
    monkeypatch.setenv(WEB_FETCH_ENDPOINT_ENV, _PROXY_ENDPOINT)
    get_mock = Mock()
    post_mock = Mock()
    monkeypatch.setattr(
        "flwr.supercore.task_process.connector.web_fetch.requests.get",
        get_mock,
    )
    monkeypatch.setattr(
        "flwr.supercore.task_process.connector.web_fetch.requests.post",
        post_mock,
    )

    with pytest.raises(WebFetchProviderError) as exc_info:
        invoke_web_fetch_provider(blocked_url)

    assert exc_info.value.code == "invalid_request"
    get_mock.assert_not_called()
    post_mock.assert_not_called()


def test_invoke_web_fetch_provider_enforces_fetch_guardrails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider should reject unsafe redirects, DNS, and oversized responses."""
    redirect_response = _Response(
        status_code=302,
        url="https://example.com",
        headers={"Location": "http://127.0.0.1/private"},
    )
    get_mock = _patch_get(monkeypatch, redirect_response)

    with pytest.raises(WebFetchProviderError) as exc_info:
        invoke_web_fetch_provider("https://example.com")

    assert exc_info.value.code == "blocked_url"
    assert get_mock.call_count == 1
    assert redirect_response.closed

    _patch_dns(monkeypatch, "127.0.0.1")
    get_mock = Mock()
    monkeypatch.setattr(
        "flwr.supercore.task_process.connector.web_fetch.requests.get",
        get_mock,
    )

    with pytest.raises(WebFetchProviderError) as exc_info:
        invoke_web_fetch_provider("https://private.example")

    assert exc_info.value.code == "blocked_url"
    get_mock.assert_not_called()

    _patch_dns(monkeypatch, "93.184.216.34")
    response = _Response(chunks=[b"x" * (1024 * 1024), b"x"])
    _patch_get(monkeypatch, response)

    with pytest.raises(WebFetchProviderError) as exc_info:
        invoke_web_fetch_provider("https://example.com")

    assert exc_info.value.code == "response_too_large"
    assert exc_info.value.status_code == 200
    assert response.closed


@pytest.mark.parametrize("blocked_url", ["https://100.64.0.1", "https://224.0.0.1"])
def test_invoke_web_fetch_provider_blocks_non_public_ip_literals(
    monkeypatch: pytest.MonkeyPatch,
    blocked_url: str,
) -> None:
    """Provider should reject non-public IP literals before fetching."""
    get_mock = Mock()
    monkeypatch.setattr(
        "flwr.supercore.task_process.connector.web_fetch.requests.get",
        get_mock,
    )

    with pytest.raises(WebFetchProviderError) as exc_info:
        invoke_web_fetch_provider(blocked_url)

    assert exc_info.value.code == "blocked_url"
    get_mock.assert_not_called()
