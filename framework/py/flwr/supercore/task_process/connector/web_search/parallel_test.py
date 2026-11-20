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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from threading import Event, Thread
from unittest.mock import Mock

import pytest
import requests
from urllib3.response import HTTPResponse

from flwr.supercore.version import package_name, package_version

from .parallel import (
    PARALLEL_MCP_URL,
    ParallelWebSearchProvider,
    _parse_results,
    _read_messages,
)


def _set_body(response: requests.Response, body: bytes) -> None:
    response.raw = HTTPResponse(body=BytesIO(body), preload_content=False)


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
            _set_body(response, b"")
            return response
        assert request.body is not None
        body = json.loads(request.body)
        method = body["method"]
        if method == "notifications/initialized":
            response.status_code = 202
            _set_body(response, b"")
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
        _set_body(response, data.encode("utf-8"))
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
    _set_body(response, json.dumps({"id": 1, **body}).encode())
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
        _set_body(result, json.dumps(body).encode())
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
    _set_body(response, data.encode("utf-8"))

    assert list(_read_messages(response)) == [{"results": [{"excerpts": [excerpt]}]}]


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
    body = line_ending.join(
        [
            *[f"data: {line}" for line in data.split("\n")],
            "",
            "",
        ]
    ).encode("utf-8")
    _set_body(response, body)
    send = Mock(return_value=response)
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)

    with pytest.raises(RuntimeError, match=message):
        ParallelWebSearchProvider().search("Flower")
    send.assert_called_once()


@pytest.mark.parametrize(
    "content_type", ["text/event-stream", "TeXt/EvEnT-StReAm ; Charset=UTF-8"]
)
@pytest.mark.parametrize("line_ending", ["\n", "\r\n", "\r", "mixed"])
@pytest.mark.parametrize("rpc_error", [False, True])
def test_search_returns_before_sse_eof(
    monkeypatch: pytest.MonkeyPatch,
    content_type: str,
    line_ending: str,
    rpc_error: bool,
) -> None:
    """A real HTTP server keeps the short SSE reply open until the client returns."""
    # pylint: disable=too-many-statements
    flushed, release, done = Event(), Event(), Event()
    outcomes: list[object] = []
    requests_seen: list[tuple[str, str | None]] = []
    excerpt = "before\u0085between\u2028another\u2029after"

    class Handler(BaseHTTPRequestHandler):
        """Serve a complete reply without closing its SSE connection."""

        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # pylint: disable=invalid-name
            """Serve MCP replies, leaving only the tools/call response open."""
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            method = body["method"]
            requests_seen.append((method, self.headers.get("User-Agent")))
            if method == "notifications/initialized":
                self.send_response(202)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            result = {
                "initialize": {"protocolVersion": "2025-03-26"},
                "tools/list": {"tools": [{"name": "web_search"}]},
                "tools/call": {
                    "structuredContent": {
                        "results": [
                            {
                                "title": "Flower",
                                "url": "https://flower.ai",
                                "excerpts": [excerpt],
                            }
                        ]
                    }
                },
            }[method]
            reply = {"id": body["id"], "result": result}
            if method == "tools/call" and rpc_error:
                reply = {"id": body["id"], "error": {"code": -32000}}
            self.send_response(200)
            if method != "tools/call":
                json_body = json.dumps(reply).encode()
                self.send_header("Content-Type", "application/json")
                self.send_header("Mcp-Session-Id", "fixture-session")
                self.send_header("Content-Length", str(len(json_body)))
                self.end_headers()
                self.wfile.write(json_body)
                return
            self.send_header("Content-Type", content_type)
            self.end_headers()
            lines = [
                ": heartbeat",
                'data: {"method":"ping"}',
                "",
                'data: {"id":999,"result":{}}',
                "",
                *[
                    "data: " + line
                    for line in json.dumps(reply, ensure_ascii=False, indent=2).split(
                        "\n"
                    )
                ],
                "",
                "",
            ]
            endings = ["\r\n", "\n", "\r"] if line_ending == "mixed" else [line_ending]
            data = "".join(
                line + endings[index % len(endings)] for index, line in enumerate(lines)
            )
            self.wfile.write(data.encode("utf-8"))
            self.wfile.flush()
            flushed.set()
            release.wait(5)
            # pylint: disable-next=attribute-defined-outside-init
            self.close_connection = True

        def do_DELETE(self) -> None:  # pylint: disable=invalid-name
            """Acknowledge MCP session cleanup while the search stream is open."""
            requests_seen.append(("DELETE", self.headers.get("User-Agent")))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        # pylint: disable-next=arguments-differ
        def log_message(self, _format: str, *args: object) -> None:
            """Keep local HTTP logs out of test output."""

    def search() -> None:
        try:
            outcomes.append(ParallelWebSearchProvider().search("Flower"))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            outcomes.append(exc)
        finally:
            done.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    server_thread.start()
    monkeypatch.setattr(
        "flwr.supercore.task_process.connector.web_search.parallel.PARALLEL_MCP_URL",
        f"http://127.0.0.1:{server.server_port}/mcp",
    )
    client_thread = Thread(target=search, daemon=True)
    client_thread.start()
    try:
        assert flushed.wait(3), "fixture did not flush the complete RPC reply"
        assert done.wait(1), "client waited for EOF after a complete RPC reply"
        if rpc_error:
            assert isinstance(outcomes[0], RuntimeError)
            assert "RPC error" in str(outcomes[0])
        else:
            assert outcomes == [
                {
                    "results": [
                        {
                            "title": "Flower",
                            "url": "https://flower.ai",
                            "snippet": excerpt,
                            "published_at": None,
                        }
                    ]
                }
            ]
        assert [method for method, _ in requests_seen] == [
            "initialize",
            "notifications/initialized",
            "tools/list",
            "tools/call",
            "DELETE",
        ]
        assert all(
            agent == f"{package_name}/{package_version}" for _, agent in requests_seen
        )
    finally:
        release.set()
        client_thread.join(timeout=3)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=3)
