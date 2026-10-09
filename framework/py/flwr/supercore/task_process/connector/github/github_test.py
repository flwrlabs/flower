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

import traceback
from collections.abc import Iterator
from typing import cast
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import requests
from mcp import types
from mcp.shared.exceptions import McpError

from flwr.supercore.typing import JSONObject, JSONValue

from .. import registry
from ..definition import ActionAccess
from ..oauth import OAuthFlow
from .definition import CONNECTOR, PROVIDER, load_connector
from .mcp import GitHubApiError

_ACCESS_TOKEN = "github-private-token-fixture"
_TOKEN_REQUEST = "flwr.supercore.task_process.connector.oauth.requests.post"
_IDENTITY_REQUEST = "flwr.supercore.task_process.connector.oauth.requests.request"


def _response(payload: object, status_code: int = 200) -> Mock:
    """Return a minimal HTTP response mock."""
    response = Mock(status_code=status_code)
    response.json.return_value = payload
    return response


@pytest.fixture(name="session")
def fixture_session() -> Iterator[AsyncMock]:
    """Mock the SDK boundary and check Flower's connection settings."""
    session = AsyncMock()
    session.__aenter__.return_value = session
    session.list_tools.return_value = types.ListToolsResult(
        tools=[types.Tool(name="FutureTool", inputSchema={"type": "object"})]
    )
    session.call_tool.return_value = types.CallToolResult(
        content=[types.TextContent(type="text", text="result")],
        structuredContent={"count": 1},
    )
    transport = MagicMock()
    transport.__aenter__.return_value = (Mock(), Mock(), Mock())
    with (
        patch("mcp.ClientSession", return_value=session),
        patch(
            "mcp.client.streamable_http.streamable_http_client",
            return_value=transport,
        ) as connect,
    ):
        yield session
    assert connect.call_args.args == ("https://api.githubcopilot.com/mcp/",)
    client = connect.call_args.kwargs["http_client"]
    assert client.headers["Authorization"] == f"Bearer {_ACCESS_TOKEN}"
    assert client.headers["X-MCP-Toolsets"] == "all"
    assert "X-MCP-Readonly" not in client.headers
    assert "X-MCP-Tools" not in client.headers
    assert client.is_closed
    assert session.initialize.await_count == connect.call_count
    assert session.__aexit__.await_count == connect.call_count


def _invoke(name: str, arguments: JSONObject) -> JSONValue:
    """Call a tool through Flower's existing GitHub registry entry."""
    return registry.invoke_connector(
        name, arguments, Mock(), {"access_token": _ACCESS_TOKEN}, {}
    )


def test_discovery_preserves_paginated_remote_catalog(session: AsyncMock) -> None:
    """Expose remote read and write tools with their original schemas."""
    schema = {"type": "object", "properties": {"data": {"type": "object"}}}
    session.list_tools.side_effect = [
        types.ListToolsResult(
            tools=[
                types.Tool(name="issue_write", description="Write", inputSchema=schema)
            ],
            nextCursor="next",
        ),
        session.list_tools.return_value,
    ]
    tools = cast(list[JSONObject], _invoke("github__discover_tools", {}))
    assert tools[0] == {
        "type": "function",
        "name": "github_issue_write",
        "description": "Write",
        "parameters": schema,
        "strict": False,
    }
    assert tools[1]["name"] == "github_futuretool"
    assert session.list_tools.await_args_list == [
        call(params=None),
        call(params=types.PaginatedRequestParams(cursor="next")),
    ]
    session.call_tool.assert_not_awaited()


@pytest.mark.parametrize("read_only", [True, False, None])
def test_catalog_builds_connection_specific_actions(
    session: AsyncMock, read_only: bool | None
) -> None:
    """Use normal actions and executors without changing the shared definition."""
    session.list_tools.return_value.tools[0].annotations = types.ToolAnnotations(
        readOnlyHint=read_only
    )
    connector = load_connector({"access_token": _ACCESS_TOKEN})
    assert connector.provider is not None
    assert connector.provider.actions[0].access == (
        ActionAccess.READ if read_only else ActionAccess.WRITE
    )
    assert set(connector.executors) == {"github_futuretool"}
    assert connector.load_connector is None
    assert not CONNECTOR.tools and not CONNECTOR.executors


@pytest.mark.parametrize("name", ["search_code", "issue_write", "FutureTool"])
def test_calls_forward_remote_names_arguments_and_results(
    session: AsyncMock, name: str
) -> None:
    """Forward advertised tools, native fields, nested arguments and MCP output."""
    session.list_tools.return_value = types.ListToolsResult(
        tools=[types.Tool(name=name, inputSchema={"type": "object"})]
    )
    arguments: JSONObject = {"perPage": 5, "data": {"items": [1, False, None]}}
    assert _invoke(f"github_{name.lower()}", arguments) == (
        session.call_tool.return_value.model_dump(
            mode="json", by_alias=True, exclude_none=True
        )
    )
    session.call_tool.assert_awaited_once_with(name, arguments)


def test_unadvertised_tool_is_not_called(session: AsyncMock) -> None:
    """Reject tools absent from the connection's catalog."""
    with pytest.raises(ValueError, match="Unsupported connector"):
        _invoke("github_unknown", {})
    session.call_tool.assert_not_awaited()


@pytest.mark.parametrize("failure", ["tool", "http", "protocol"])
def test_errors_do_not_expose_credentials(session: AsyncMock, failure: str) -> None:
    """Keep connector failures readable without leaking tokens or raw exceptions."""
    if failure == "tool":
        session.call_tool.return_value = types.CallToolResult(
            content=[types.TextContent(type="text", text=f"Denied {_ACCESS_TOKEN}")],
            isError=True,
        )
    elif failure == "http":
        session.call_tool.side_effect = httpx.HTTPStatusError(
            f"Forbidden {_ACCESS_TOKEN}",
            request=httpx.Request("POST", "https://api.githubcopilot.com/mcp/"),
            response=httpx.Response(403),
        )
    else:
        session.call_tool.side_effect = ExceptionGroup(
            "MCP request failed",
            [
                McpError(
                    types.ErrorData(
                        code=-32602, message=f"Invalid arguments {_ACCESS_TOKEN}"
                    )
                )
            ],
        )
    with pytest.raises(GitHubApiError) as error:
        _invoke("github_futuretool", {})
    assert (
        error.value.code
        == {
            "tool": "tool_error",
            "http": "request_failed",
            "protocol": "request_failed",
        }[failure]
    )
    assert _ACCESS_TOKEN not in "".join(traceback.format_exception(error.value))
    assert (
        error.value.message
        == {
            "tool": "Denied [REDACTED]",
            "http": "Forbidden [REDACTED]",
            "protocol": "Invalid arguments [REDACTED]",
        }[failure]
    )
    if failure == "http":
        assert error.value.status_code == 403


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
