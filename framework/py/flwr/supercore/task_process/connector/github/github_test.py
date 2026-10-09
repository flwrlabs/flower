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
"""Tests for the GitHub connector."""

import json
import sys
import traceback
from typing import cast
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import requests
from mcp import types

from flwr.supercore.typing import JSONObject, JSONValue

from .. import registry
from ..oauth import OAuthFlow
from . import mcp
from .actions import DISCOVERY_TOOL
from .definition import PROVIDER
from .mcp import GitHubApiError

_ACCESS_TOKEN = "github-private-token-fixture"
_AsyncClient = httpx.AsyncClient
_TOKEN_REQUEST = "flwr.supercore.task_process.connector.oauth.requests.post"
_IDENTITY_REQUEST = "flwr.supercore.task_process.connector.oauth.requests.request"


def _response(payload: object, status_code: int = 200) -> Mock:
    """Return a minimal HTTP response mock."""
    response = Mock(status_code=status_code)
    response.json.return_value = payload
    return response


class _MCPServer:  # pylint: disable=too-many-instance-attributes
    """Exercise the SDK's HTTP transport with JSON-RPC responses."""

    def __init__(self) -> None:
        self.requests: list[JSONObject] = []
        self.tools = ["search_code", "get_file_contents"]
        self.schemas: dict[str, JSONObject] = {}
        self.next_tools: list[str] | None = None
        self.result: JSONObject = {
            "content": [{"type": "text", "text": "result"}],
            "isError": False,
        }
        self.sse = False
        self.status_code = 200
        self.transport_error: Exception | None = None
        self.rpc_error = False
        self.closed = False
        self.repeat_cursor = False

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Respond to initialization, notifications, discovery and calls."""
        assert str(request.url) == "https://api.githubcopilot.com/mcp/"
        assert request.headers["Authorization"] == f"Bearer {_ACCESS_TOKEN}"
        assert request.headers["X-MCP-Toolsets"] == "all"
        assert "X-MCP-Readonly" not in request.headers
        assert "X-MCP-Tools" not in request.headers
        if request.method == "GET":
            return httpx.Response(405)
        if request.method == "DELETE":
            assert request.headers["mcp-session-id"] == "test-session"
            self.closed = True
            return httpx.Response(204)
        payload = cast(JSONObject, json.loads(request.content))
        self.requests.append(payload)
        if self.transport_error:
            raise self.transport_error
        if self.status_code != 200:
            return httpx.Response(self.status_code, text=_ACCESS_TOKEN)
        method = payload["method"]
        if method == "initialize":
            result: JSONObject = {
                "protocolVersion": types.LATEST_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "github", "version": "test"},
            }
        else:
            assert request.headers["mcp-session-id"] == "test-session"
            assert request.headers["mcp-protocol-version"]
            if method == "notifications/initialized":
                return httpx.Response(202)
            if method == "tools/list":
                params = cast(JSONObject, payload.get("params", {}))
                names = self.next_tools if params.get("cursor") else self.tools
                result = {
                    "tools": [
                        {
                            "name": name,
                            "description": f"GitHub {name}",
                            "inputSchema": self.schemas.get(name, {"type": "object"}),
                        }
                        for name in names or []
                    ]
                }
                if self.repeat_cursor or (
                    self.next_tools is not None and not params.get("cursor")
                ):
                    result["nextCursor"] = "next"
            else:
                assert method == "tools/call"
                result = self.result
        reply: JSONObject = {"jsonrpc": "2.0", "id": payload["id"], "result": result}
        if self.rpc_error and method == "tools/call":
            del reply["result"]
            reply["error"] = {"code": -32602, "message": _ACCESS_TOKEN}
        headers = {"mcp-session-id": "test-session"}
        if self.sse:
            headers["Content-Type"] = "text/event-stream"
            return httpx.Response(
                200,
                content=f"event: message\ndata: {json.dumps(reply)}\n\n",
                headers=headers,
            )
        return httpx.Response(200, json=reply, headers=headers)


@pytest.fixture(name="server")
def fixture_server(monkeypatch: pytest.MonkeyPatch) -> _MCPServer:
    """Replace network access while retaining the real MCP SDK."""
    remote = _MCPServer()

    def make_client(
        *, headers: dict[str, str], timeout: float, follow_redirects: bool
    ) -> httpx.AsyncClient:
        assert timeout == 30.0
        assert not follow_redirects
        return _AsyncClient(
            headers=headers,
            timeout=timeout,
            follow_redirects=follow_redirects,
            transport=httpx.MockTransport(remote.handle),
        )

    monkeypatch.setattr(httpx, "AsyncClient", make_client)
    return remote


def _invoke(name: str, arguments: JSONObject) -> JSONValue:
    """Call a tool through Flower's existing GitHub registry entry."""
    return registry.invoke_connector(
        name, arguments, Mock(), {"access_token": _ACCESS_TOKEN}, {}
    )


@pytest.mark.parametrize("sse", [False, True])
def test_get_file_contents_returns_mcp_content(server: _MCPServer, sse: bool) -> None:
    """Preserve text, resources and structured data in JSON and SSE replies."""
    server.sse = sse
    server.result = {
        "content": [
            {"type": "text", "text": "File contents"},
            {
                "type": "resource",
                "resource": {
                    "uri": "repo://acme/repo/src/app.py",
                    "text": "print('hi')",
                },
            },
        ],
        "structuredContent": {"path": "src/app.py"},
        "isError": False,
    }
    result = _invoke(
        "github_get_file_contents",
        {"owner": "acme", "repo": "repo", "path": "/src/app.py/", "ref": "main"},
    )
    assert result == server.result
    assert [request["method"] for request in server.requests] == [
        "initialize",
        "notifications/initialized",
        "tools/list",
        "tools/call",
    ]
    assert server.requests[-1]["params"] == {
        "name": "get_file_contents",
        "arguments": {
            "owner": "acme",
            "repo": "repo",
            "path": "/src/app.py/",
            "ref": "main",
        },
    }
    assert server.closed


def test_github_search_translates_pagination(server: _MCPServer) -> None:
    """Keep Flower's per_page argument and send MCP's numeric perPage."""
    result = _invoke(
        "github_search_code",
        {
            "query": "Flower repo:acme/repo",
            "sort": "indexed",
            "order": "desc",
            "per_page": 5,
            "page": 101,
        },
    )
    assert result == server.result
    assert server.requests[-1]["params"] == {
        "name": "search_code",
        "arguments": {
            "query": "Flower repo:acme/repo",
            "sort": "indexed",
            "order": "desc",
            "perPage": 5,
            "page": 101,
        },
    }


@pytest.mark.parametrize(
    "arguments",
    [
        {"owner": "acme", "repo": "repo"},
        {"owner": "acme", "repo": "repo", "path": "/", "fields": ["name", "type"]},
        {"owner": "acme", "repo": "repo", "path": "file", "sha": "abc123"},
    ],
)
def test_file_arguments_use_full_remote_schema(
    server: _MCPServer, arguments: JSONObject
) -> None:
    """Support directories, omitted paths, commit SHAs and field selection."""
    _invoke("github_get_file_contents", arguments)
    assert server.requests[-1]["params"] == {
        "name": "get_file_contents",
        "arguments": arguments,
    }


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": "Flower", "per_page": 101},
        {"query": "Flower", "per_page": True},
        {"query": "Flower", "per_page": 0},
        {"query": "Flower", "per_page": 5, "perPage": 10},
    ],
)
def test_invalid_search_arguments_never_reach_mcp(
    server: _MCPServer, arguments: JSONObject
) -> None:
    """Reject invalid or ambiguous legacy pagination before contacting GitHub."""
    with pytest.raises(ValueError):
        _invoke("github_search_code", arguments)
    assert not server.requests


def test_discovery_follows_pagination(server: _MCPServer) -> None:
    """Find the requested tool even when GitHub paginates tools/list."""
    server.tools = []
    server.next_tools = ["search_code"]
    _invoke("github_search_code", {"query": "Flower"})
    assert server.requests[-2]["params"] == {"cursor": "next"}
    assert server.closed


def test_unavailable_tool_is_not_called(server: _MCPServer) -> None:
    """Fail safely when a connection does not have the requested tool."""
    server.tools = []
    with pytest.raises(GitHubApiError) as error:
        _invoke("github_search_code", {"query": "Flower"})
    assert error.value.code == "tool_unavailable"
    assert all(request["method"] != "tools/call" for request in server.requests)
    assert server.closed


@pytest.mark.parametrize("status", [401, 403, 429])
def test_http_errors_are_secret_safe(server: _MCPServer, status: int) -> None:
    """Keep HTTP status codes and never include response bodies or credentials."""
    server.status_code = status
    with pytest.raises(GitHubApiError) as error:
        _invoke("github_search_code", {"query": "Flower"})
    assert error.value.status_code == status
    assert _ACCESS_TOKEN not in "".join(traceback.format_exception(error.value))


@pytest.mark.parametrize(
    "failure", [RuntimeError(_ACCESS_TOKEN), httpx.ConnectTimeout(_ACCESS_TOKEN)]
)
def test_transport_errors_are_secret_safe(
    server: _MCPServer, failure: Exception
) -> None:
    """Discard transport diagnostic text and exception context."""
    server.transport_error = failure
    with pytest.raises(GitHubApiError) as error:
        _invoke("github_search_code", {"query": "Flower"})
    assert _ACCESS_TOKEN not in "".join(traceback.format_exception(error.value))


def test_mcp_protocol_errors_are_secret_safe(server: _MCPServer) -> None:
    """Discard raw JSON-RPC diagnostic text."""
    server.rpc_error = True
    with pytest.raises(GitHubApiError) as error:
        _invoke("github_search_code", {"query": "Flower"})
    assert error.value.code == "request_failed"
    assert _ACCESS_TOKEN not in "".join(traceback.format_exception(error.value))
    assert server.closed


def test_tool_errors_are_connector_errors(server: _MCPServer) -> None:
    """Report MCP isError results through Flower's connector error channel."""
    server.result = {
        "content": [{"type": "text", "text": f"Denied {_ACCESS_TOKEN}"}],
        "isError": True,
    }
    with pytest.raises(GitHubApiError) as error:
        _invoke("github_search_code", {"query": "Flower"})
    assert error.value.code == "tool_error"
    assert error.value.message == "Denied [REDACTED]"
    assert _ACCESS_TOKEN not in "".join(traceback.format_exception(error.value))
    assert server.closed


@pytest.mark.usefixtures("server")
def test_total_timeout_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bound initialization, discovery, tool calls and cleanup together."""
    monkeypatch.setattr(mcp, "_CALL_TIMEOUT", 0)
    with pytest.raises(GitHubApiError) as error:
        _invoke("github_search_code", {"query": "Flower"})
    assert error.value.code == "timeout"


def test_full_catalog_preserves_github_identity(server: _MCPServer) -> None:
    """Expose new read and write tools through the same GitHub connector."""
    assert PROVIDER.ref == "github"
    assert PROVIDER.display_name == "GitHub"
    server.tools += ["issue_write", "future_tool"]
    tools = _invoke(DISCOVERY_TOOL, {})
    assert isinstance(tools, list)
    assert [cast(JSONObject, tool)["name"] for tool in tools] == [
        f"github_{name}" for name in server.tools
    ]
    assert all(request["method"] != "tools/call" for request in server.requests)
    assert server.closed


@pytest.mark.parametrize("name", ["issue_write", "merge_pull_request", "future_tool"])
def test_dynamic_read_and_write_calls_are_forwarded(
    server: _MCPServer, name: str
) -> None:
    """Support any advertised tool and forward nested arguments unchanged."""
    server.tools += [name]
    arguments: JSONObject = {"owner": "acme", "data": {"items": [1, False, None]}}
    result = _invoke(f"github_{name}", arguments)
    assert result == server.result
    calls = [
        request for request in server.requests if request["method"] == "tools/call"
    ]
    assert len(calls) == 1
    assert calls[0]["params"] == {"name": name, "arguments": arguments}
    assert server.closed


def test_unknown_remote_tool_cannot_be_called(server: _MCPServer) -> None:
    """Reject a tool absent from the authenticated server's catalog."""
    with pytest.raises(GitHubApiError, match="tool_unavailable"):
        _invoke("github_not_advertised", {})
    assert all(request["method"] != "tools/call" for request in server.requests)
    assert server.closed


def test_discovery_preserves_remote_schemas(server: _MCPServer) -> None:
    """Use GitHub's description and schema rather than a Flower action list."""
    server.tools = ["issue_write"]
    server.next_tools = ["future_tool"]
    schema: JSONObject = {
        "type": "object",
        "properties": {"data": {"oneOf": [{"type": "string"}, {"type": "array"}]}},
        "required": ["data"],
        "additionalProperties": False,
    }
    server.schemas["issue_write"] = schema
    tools = _invoke(DISCOVERY_TOOL, {})
    assert isinstance(tools, list)
    assert tools[0] == {
        "type": "function",
        "name": "github_issue_write",
        "description": "GitHub issue_write",
        "parameters": schema,
        "strict": False,
    }
    assert cast(JSONObject, tools[1])["name"] == "github_future_tool"
    assert server.closed


@pytest.mark.parametrize("tools", [["same", "same"], ["same", "SAME"]])
def test_ambiguous_tool_names_are_rejected(
    server: _MCPServer, tools: list[str]
) -> None:
    """Prevent duplicate model names after applying the GitHub namespace."""
    server.tools = tools
    with pytest.raises(GitHubApiError, match="invalid_response"):
        _invoke(DISCOVERY_TOOL, {})
    assert server.closed


def test_repeated_discovery_cursor_is_rejected(server: _MCPServer) -> None:
    """Stop a malformed tools/list pagination cycle without timing out."""
    server.tools = []
    server.next_tools = []
    server.repeat_cursor = True
    with pytest.raises(GitHubApiError, match="invalid_response"):
        _invoke(DISCOVERY_TOOL, {})
    assert server.closed


def test_remote_name_casing_is_preserved_on_call(server: _MCPServer) -> None:
    """Match normalized model names while calling the server's original name."""
    server.tools = ["FutureTool"]
    _invoke("github_futuretool", {})
    assert server.requests[-1]["params"] == {"name": "FutureTool", "arguments": {}}


@pytest.mark.parametrize("credentials", [{}, {"access_token": ""}, {"access_token": 1}])
def test_invalid_credentials_never_reach_mcp(
    server: _MCPServer, credentials: JSONObject
) -> None:
    """Require a token before contacting GitHub."""
    with pytest.raises(GitHubApiError, match="invalid_credentials"):
        mcp.call_tool("search_code", {"query": "Flower"}, credentials)
    assert not server.requests


def test_missing_mcp_dependency_is_actionable(server: _MCPServer) -> None:
    """Report a missing SDK without attempting a remote call."""
    with patch.dict(sys.modules, {"mcp": None}), pytest.raises(GitHubApiError) as error:
        _invoke("github_search_code", {"query": "Flower"})
    assert error.value.code == "missing_dependency"
    assert "MCP SDK" in str(error.value)
    assert not server.requests


def test_github_oauth_requests_no_scope() -> None:
    """OAuth should request and accept only scope-free credentials."""
    flow = OAuthFlow(
        PROVIDER,
        client_id="client",
        client_secret="secret",
        redirect_uri="https://example.com/callback",
    )
    url = flow.build_authorization_url(
        redirect_uri="https://example.com/callback",
        state="state",
        pkce_challenge="challenge",
    )
    assert "scope" not in parse_qs(urlparse(url).query)
    token_response = _response(
        {"access_token": "token", "token_type": "bearer", "scope": ""}
    )
    with (
        patch(_TOKEN_REQUEST, return_value=token_response),
        patch(
            _IDENTITY_REQUEST, return_value=_response({"login": "octocat"})
        ) as identity,
    ):
        credentials, config = flow.exchange_code(
            code="code",
            redirect_uri="https://example.com/callback",
            pkce_verifier="verifier",
        )
    assert credentials == {"access_token": "token", "token_type": "bearer"}
    assert config == {"display_name": "GitHub · octocat"}
    assert identity.call_args.args == ("GET", "https://api.github.com/user")
    assert identity.call_args.kwargs["headers"]["Authorization"] == "Bearer token"

    with (
        patch(_TOKEN_REQUEST, return_value=token_response),
        patch(_IDENTITY_REQUEST, side_effect=requests.Timeout),
    ):
        _, config = flow.exchange_code(
            code="code",
            redirect_uri="https://example.com/callback",
            pkce_verifier="verifier",
        )
    assert not config

    token_response.json.return_value["scope"] = "repo"
    with (
        patch(_TOKEN_REQUEST, return_value=token_response),
        pytest.raises(RuntimeError),
    ):
        flow.exchange_code(
            code="code",
            redirect_uri="https://example.com/callback",
            pkce_verifier="verifier",
        )
