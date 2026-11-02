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
"""Essential tests for Notion MCP and OAuth."""

import traceback
from typing import cast
from unittest.mock import AsyncMock, Mock, call, patch
from urllib.parse import parse_qs, urlparse

import pytest
from mcp import ClientSession, types

from flwr.supercore.typing import JSONObject

from .. import registry
from ..oauth import OAuthFlow
from . import mcp
from .definition import PROVIDER

_CREDENTIALS: JSONObject = {"access_token": "private-token"}


def test_discovery_and_execution() -> None:
    """Preserve paginated schemas and call once without a trailing catalog lookup."""
    schema = {"type": "object", "properties": {"title": {"type": "string"}}}
    rpc = AsyncMock(
        side_effect=[
            types.ListToolsResult(
                tools=[types.Tool(name="FutureTool", inputSchema=schema)],
                nextCursor="next",
            ),
            types.ListToolsResult(
                tools=[types.Tool(name="notion-search", inputSchema=schema)]
            ),
        ]
    )
    with (
        patch.object(ClientSession, "send_request", rpc),
        patch.object(mcp, "_session") as open_session,
    ):
        open_session.return_value.__aenter__.side_effect = lambda: ClientSession(
            Mock(), Mock()
        )
        tools = cast(
            list[JSONObject],
            registry.invoke_connector(
                "notion__discover_tools", {}, Mock(), _CREDENTIALS, {}
            ),
        )
        assert [tool["name"] for tool in tools] == [
            "notion_FutureTool",
            "notion_notion-search",
        ]
        assert tools[0]["parameters"] == schema
        assert rpc.await_args_list[1].args[0].root.params.cursor == "next"

        output = types.CallToolResult(
            content=[types.TextContent(type="text", text="done")]
        )
        rpc.reset_mock(side_effect=True)
        rpc.side_effect = [output, RuntimeError("Catalog unavailable")]
        result = registry.invoke_connector(
            "notion_FutureTool", {"title": "New page"}, Mock(), _CREDENTIALS, {}
        )
    assert result == output.model_dump(mode="json", by_alias=True, exclude_none=True)
    rpc.assert_awaited_once()
    request = rpc.await_args_list[0].args[0].root
    assert request.params.name == "FutureTool"
    assert request.params.arguments == {"title": "New page"}


def test_fallback_does_not_retry_failed_tool() -> None:
    """Fall back during setup, then preserve a redacted tool failure without retry."""
    session = Mock(spec=ClientSession)
    session.send_request.return_value = types.CallToolResult(
        isError=True,
        content=[types.TextContent(type="text", text="Denied private-token")],
    )
    with patch.object(mcp, "_initialized_session") as connect:
        connect.return_value.__aenter__.side_effect = [
            RuntimeError("HTTP unavailable"),
            session,
        ]
        with pytest.raises(mcp.NotionApiError) as error:
            registry.invoke_connector("notion_FutureTool", {}, Mock(), _CREDENTIALS, {})
    assert connect.call_args_list == [
        call("private-token", use_sse=False),
        call("private-token", use_sse=True),
    ]
    session.send_request.assert_awaited_once()
    assert error.value.code == "tool_error"
    assert error.value.message == "Denied [REDACTED]"
    assert "private-token" not in "".join(traceback.format_exception(error.value))


def test_oauth_resource_exchange_and_refresh() -> None:
    """Bind Notion's authorization, code exchange, and refresh to its resource."""
    flow = OAuthFlow(
        PROVIDER,
        client_id="client",
        client_secret="secret",
        redirect_uri="https://example.com/callback",
    )
    url = urlparse(
        flow.build_authorization_url(
            redirect_uri="https://example.com/callback",
            state="state",
            pkce_challenge="challenge",
        )
    )
    query = parse_qs(url.query)
    assert query["resource"] == ["https://mcp.notion.com/mcp"]
    assert query["code_challenge_method"] == ["S256"]
    response = Mock(status_code=200)
    response.json.return_value = {"access_token": "token", "token_type": "bearer"}
    with patch(
        "flwr.supercore.task_process.connector.oauth.requests.post",
        return_value=response,
    ) as post:
        credentials, _ = flow.exchange_code(
            code="code",
            redirect_uri="https://example.com/callback",
            pkce_verifier="verifier",
        )
        flow.refresh_credentials(
            {**credentials, "refresh_token": "refresh", "expires_at": 0}
        )
    exchange, refresh = post.call_args_list
    for request in (exchange, refresh):
        assert request.args == ("https://mcp.notion.com/token",)
        assert request.kwargs["data"]["resource"] == "https://mcp.notion.com/mcp"
    assert exchange.kwargs["data"]["code_verifier"] == "verifier"
    assert refresh.kwargs["data"]["grant_type"] == "refresh_token"
    assert credentials["access_token"] == "token"
