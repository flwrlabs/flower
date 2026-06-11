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

from collections.abc import Iterator
from dataclasses import dataclass, field
from unittest.mock import Mock

import pytest

from .web_fetch import WebFetchProviderError, invoke_web_fetch_provider


@dataclass
class _Response:
    status_code: int = 200
    url: str = "https://example.com/final"
    headers: dict[str, str] = field(
        default_factory=lambda: {"Content-Type": "text/html; charset=utf-8"}
    )
    body: bytes = b"<html><body><main>Hello</main></body></html>"
    text: str = ""
    encoding: str | None = "utf-8"
    apparent_encoding: str | None = "utf-8"
    closed: bool = False

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        """Return the mocked response body in chunks."""
        del chunk_size
        yield self.body

    def close(self) -> None:
        """Record that the response was closed."""
        self.closed = True


def _patch_get(monkeypatch: pytest.MonkeyPatch, response: _Response) -> Mock:
    get_mock = Mock(return_value=response)
    monkeypatch.setattr(
        "flwr.supercore.task_process.connector.web_fetch.requests.get",
        get_mock,
    )
    return get_mock


def test_invoke_web_fetch_provider_extracts_markdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-raw requests should return trafilatura-extracted markdown."""
    response = _Response()
    get_mock = _patch_get(monkeypatch, response)
    extract_mock = Mock(return_value="# Hello")
    monkeypatch.setattr(
        "flwr.supercore.task_process.connector.web_fetch._extract_markdown",
        extract_mock,
    )

    result = invoke_web_fetch_provider({"url": "https://example.com"})

    assert result == {
        "object": "web_fetch.response",
        "status": "completed",
        "url": "https://example.com",
        "final_url": "https://example.com/final",
        "status_code": 200,
        "content_type": "text/html; charset=utf-8",
        "content": "# Hello",
        "start_index": 0,
        "truncated": False,
        "next_start_index": None,
    }
    get_mock.assert_called_once_with(
        "https://example.com",
        headers={"User-Agent": "FlowerWebFetch/1.0"},
        timeout=30.0,
        stream=True,
    )
    extract_mock.assert_called_once_with(
        "<html><body><main>Hello</main></body></html>",
        "https://example.com/final",
    )
    assert response.closed


def test_invoke_web_fetch_provider_returns_raw_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw requests should bypass trafilatura extraction."""
    response = _Response(body=b"plain text")
    _patch_get(monkeypatch, response)
    extract_mock = Mock(side_effect=AssertionError("extract should not be called"))
    monkeypatch.setattr(
        "flwr.supercore.task_process.connector.web_fetch._extract_markdown",
        extract_mock,
    )

    result = invoke_web_fetch_provider(
        {"url": "https://example.com/plain.txt", "raw": True}
    )

    assert result["content"] == "plain text"
    assert result["content_type"] == "text/html; charset=utf-8"
    extract_mock.assert_not_called()
    assert response.closed


def test_invoke_web_fetch_provider_truncates_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider responses should support character-index pagination."""
    _patch_get(monkeypatch, _Response())
    monkeypatch.setattr(
        "flwr.supercore.task_process.connector.web_fetch._extract_markdown",
        Mock(return_value="abcdef"),
    )

    result = invoke_web_fetch_provider(
        {
            "url": "https://example.com",
            "start_index": 2,
            "max_length": 3,
        }
    )

    assert result["content"] == "cde"
    assert result["start_index"] == 2
    assert result["truncated"] is True
    assert result["next_start_index"] == 5


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({}, "invalid_request"),
        ({"url": ""}, "invalid_request"),
        ({"url": "file:///etc/passwd"}, "invalid_request"),
        ({"url": "https://localhost"}, "blocked_url"),
        ({"url": "https://127.0.0.1"}, "blocked_url"),
        ({"url": "https://example.com", "max_length": 0}, "invalid_request"),
        ({"url": "https://example.com", "start_index": -1}, "invalid_request"),
        ({"url": "https://example.com", "raw": "false"}, "invalid_request"),
    ],
)
def test_invoke_web_fetch_provider_validates_request(
    payload: dict[str, object],
    code: str,
) -> None:
    """Malformed provider requests should raise typed provider errors."""
    with pytest.raises(WebFetchProviderError) as exc_info:
        invoke_web_fetch_provider(payload)  # type: ignore[arg-type]

    assert exc_info.value.code == code


def test_invoke_web_fetch_provider_raises_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP error responses should become provider errors."""
    response = _Response(status_code=404, body=b"not found")
    _patch_get(monkeypatch, response)

    with pytest.raises(WebFetchProviderError) as exc_info:
        invoke_web_fetch_provider({"url": "https://example.com/missing"})

    assert exc_info.value.code == "http_error"
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "not found"
    assert response.closed


def test_invoke_web_fetch_provider_validates_redirect_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redirect targets should pass the same URL safety checks."""
    response = _Response(url="https://127.0.0.1/final")
    _patch_get(monkeypatch, response)

    with pytest.raises(WebFetchProviderError) as exc_info:
        invoke_web_fetch_provider({"url": "https://example.com"})

    assert exc_info.value.code == "blocked_url"
    assert response.closed


def test_invoke_web_fetch_provider_enforces_response_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Response bodies should be bounded by environment configuration."""
    monkeypatch.setenv("FLWR_WEB_FETCH_MAX_RESPONSE_BYTES", "4")
    response = _Response(body=b"12345")
    _patch_get(monkeypatch, response)

    with pytest.raises(WebFetchProviderError) as exc_info:
        invoke_web_fetch_provider({"url": "https://example.com"})

    assert exc_info.value.code == "response_too_large"
    assert exc_info.value.status_code == 200
    assert response.closed


def test_invoke_web_fetch_provider_uses_environment_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider environment settings should affect fetch and default chunking."""
    monkeypatch.setenv("FLWR_WEB_FETCH_TIMEOUT", "7")
    monkeypatch.setenv("FLWR_WEB_FETCH_USER_AGENT", "TestAgent/1.0")
    monkeypatch.setenv("FLWR_WEB_FETCH_MAX_LENGTH", "4")
    get_mock = _patch_get(monkeypatch, _Response())
    monkeypatch.setattr(
        "flwr.supercore.task_process.connector.web_fetch._extract_markdown",
        Mock(return_value="abcdef"),
    )

    result = invoke_web_fetch_provider({"url": "https://example.com"})

    assert result["content"] == "abcd"
    assert result["truncated"] is True
    assert result["next_start_index"] == 4
    assert get_mock.call_args.kwargs["headers"] == {"User-Agent": "TestAgent/1.0"}
    assert get_mock.call_args.kwargs["timeout"] == 7.0
