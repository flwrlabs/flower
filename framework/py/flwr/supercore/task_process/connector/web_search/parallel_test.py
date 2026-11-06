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
"""Tests for the anonymous Parallel MCP adapter (synthetic responses)."""

import json

# Real Response objects exercise requests encoding and redirect handling.
# pylint: disable=protected-access
from unittest.mock import Mock

import pytest
import requests

from flwr.supercore.version import package_name, package_version

from .parallel import (
    PARALLEL_MCP_URL,
    ParallelWebSearchProvider,
    _parse_results,
    _read_messages,
)


@pytest.mark.parametrize("sse", [True, False])
@pytest.mark.parametrize("structured", [True, False])
def test_search_mcp_lifecycle(
    monkeypatch: pytest.MonkeyPatch, sse: bool, structured: bool
) -> None:
    """Both MCP response encodings preserve excerpts and session headers."""
    requests_seen: list[requests.PreparedRequest] = []
    payload = {
        "results": [
            {
                "title": " Flower ",
                "url": " https://flower.ai ",
                "excerpts": [" Fédéré ", " Second passage "],
                "publish_date": " 2026-01-01 ",
            }
        ]
    }

    def send(
        _adapter: object, request: requests.PreparedRequest, **_kwargs: object
    ) -> requests.Response:
        requests_seen.append(request)
        response = requests.Response()
        response.status_code = 200
        assert request.url is not None
        response.url = request.url
        response.request = request
        if request.method == "DELETE":
            response._content = b""
            return response
        assert request.body is not None
        body = json.loads(request.body)
        method = body["method"]
        if method == "notifications/initialized":
            response.status_code = 202
            response._content = b""
            return response
        result = {
            "initialize": {"protocolVersion": "2025-03-26"},
            "tools/list": {"tools": [{"name": "web_search"}]},
            "tools/call": (
                {"structuredContent": payload}
                if structured
                else {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(payload, ensure_ascii=False),
                        }
                    ]
                }
            ),
        }[method]
        data = json.dumps(
            {"jsonrpc": "2.0", "id": body["id"], "result": result}, ensure_ascii=False
        )
        if sse:
            response.headers["Content-Type"] = "text/event-stream"
            # Include a comment and unrelated notification before the reply.
            data = (
                ': ping\r\n\r\ndata: {"method":"ping"}\r\n\r\nevent: message\r\ndata: '
                + data
                + "\r\n\r\n"
            )
        else:
            response.headers["Content-Type"] = "application/json"
        response._content = data.encode("utf-8")
        if method == "initialize":
            response.headers["Mcp-Session-Id"] = "test-session"
        if method == "tools/call":
            assert body["params"] == {
                "name": "web_search",
                "arguments": {"objective": "Flower", "search_queries": ["Flower"]},
            }
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    assert ParallelWebSearchProvider().search(" Flower ") == {
        "results": [
            {
                "title": "Flower",
                "url": "https://flower.ai",
                "snippet": "Fédéré\n\nSecond passage",
                "published_at": "2026-01-01",
            }
        ]
    }
    assert [r.method for r in requests_seen] == ["POST"] * 4 + ["DELETE"]
    for index, request in enumerate(requests_seen):
        assert request.url == PARALLEL_MCP_URL
        assert request.headers["User-Agent"] == f"{package_name}/{package_version}"
        assert "Authorization" not in request.headers
        if index:
            assert request.headers["Mcp-Session-Id"] == "test-session"
            assert request.headers["MCP-Protocol-Version"] == "2025-03-26"


@pytest.mark.parametrize(
    "body,status",
    [
        ({"error": {"code": -32000}}, 200),
        ({"result": []}, 200),
        ({"result": {}}, 401),
        ({"result": {}}, 429),
        ({"result": {}}, 500),
        ({"result": {}}, 307),
    ],
)
def test_transport_errors_are_not_retried(
    monkeypatch: pytest.MonkeyPatch, body: dict[str, object], status: int
) -> None:
    """HTTP, RPC and redirect errors stop without another provider or request."""
    response = requests.Response()
    response.status_code = status
    response.headers["Location"] = "https://example.com/redirect"
    response._content = json.dumps({"id": 1, **body}).encode()
    send = Mock(return_value=response)
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    with pytest.raises(RuntimeError, match="parallel MCP"):
        ParallelWebSearchProvider().search("Flower")
    send.assert_called_once()


def test_empty_query_does_not_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blank queries fail before network activity."""
    send = Mock()
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    with pytest.raises(ValueError, match="non-empty"):
        ParallelWebSearchProvider().search(" ")
    send.assert_not_called()


def test_parse_results() -> None:
    """Malformed entries are skipped and absent optional fields remain usable."""
    assert _parse_results(
        {"results": [None, {}, {"url": " "}, {"url": "https://flower.ai"}]}
    ) == [
        {
            "title": "https://flower.ai",
            "url": "https://flower.ai",
            "snippet": None,
            "published_at": None,
        }
    ]
    with pytest.raises(RuntimeError, match="results list"):
        _parse_results({})


@pytest.mark.parametrize(
    "reply",
    [
        {"error": {"code": -32000}},
        {"result": {"isError": True}},
        {"result": {"content": [{"type": "text", "text": "invalid JSON"}]}},
        {"result": {"structuredContent": {}}},
    ],
)
def test_tool_failures_close_session(
    monkeypatch: pytest.MonkeyPatch, reply: dict[str, object]
) -> None:
    """A failed tool call is surfaced and the MCP session is still terminated."""

    def response(body: object) -> requests.Response:
        result = requests.Response()
        result.status_code = 200
        result._content = json.dumps(body).encode()
        return result

    initialized = response({"id": 1, "result": {"protocolVersion": "2025-03-26"}})
    initialized.headers["Mcp-Session-Id"] = "test-session"
    send = Mock(
        side_effect=[
            initialized,
            response({}),
            response({"id": 2, "result": {"tools": [{"name": "web_search"}]}}),
            response({"id": 3, **reply}),
            response({}),
        ]
    )
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    with pytest.raises(RuntimeError, match="parallel MCP"):
        ParallelWebSearchProvider().search("Flower")
    assert send.call_count == 5
    assert send.call_args.args[0].method == "DELETE"


def test_network_error_is_surfaced(monkeypatch: pytest.MonkeyPatch) -> None:
    """Network failure is reported without a retry."""
    send = Mock(side_effect=requests.Timeout())
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    with pytest.raises(RuntimeError, match="request failed"):
        ParallelWebSearchProvider().search("Flower")
    send.assert_called_once()


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
@pytest.mark.parametrize("line_ending", ["\n", "\r\n", "\r", None])
@pytest.mark.parametrize("multiline", [False, True])
def test_response_preserves_unicode_excerpts(
    separator: str, line_ending: str | None, multiline: bool
) -> None:
    """Only protocol line endings separate SSE fields, even inside UTF-8 JSON."""
    excerpt = "before" + separator + "after"
    payload = {"results": [{"excerpts": [excerpt]}]}
    data = json.dumps(payload, ensure_ascii=False, indent=2 if multiline else None)
    assert separator in data
    response = requests.Response()
    if line_ending is None:
        response.headers["Content-Type"] = "application/json"
    else:
        response.headers["Content-Type"] = "text/event-stream"
        data = line_ending.join(
            [
                ": heartbeat",
                "event: message",
                *[f"data: {line}" for line in data.split("\n")],
                "",
                "",
            ]
        )
    response._content = data.encode("utf-8")

    assert _read_messages(response) == [{"results": [{"excerpts": [excerpt]}]}]


@pytest.mark.parametrize("line_ending", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize(
    "payload,message",
    [
        ({"id": 1, "error": {"code": -32000}}, "RPC error"),
        ({"id": 2, "result": {}}, "no matching result"),
        (None, "invalid JSON"),
    ],
)
def test_sse_errors_are_surfaced(
    monkeypatch: pytest.MonkeyPatch,
    line_ending: str,
    payload: dict[str, object] | None,
    message: str,
) -> None:
    """SSE RPC, mismatched reply and JSON errors propagate for each line ending."""
    data = json.dumps(payload, indent=2) if payload is not None else "{invalid-json"
    response = requests.Response()
    response.status_code = 200
    response.headers["Content-Type"] = "text/event-stream"
    response._content = line_ending.join(
        [
            *[f"data: {line}" for line in data.split("\n")],
            "",
            "",
        ]
    ).encode("utf-8")
    send = Mock(return_value=response)
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)

    with pytest.raises(RuntimeError, match=message):
        ParallelWebSearchProvider().search("Flower")
    send.assert_called_once()
