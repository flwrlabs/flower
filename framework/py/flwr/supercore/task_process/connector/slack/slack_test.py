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
"""Tests for Slack MCP discovery and execution."""

import traceback
from typing import cast
from unittest.mock import Mock, call, create_autospec, patch

import httpx
import pytest
from mcp import ClientSession
from mcp.shared.exceptions import McpError
from mcp.types import (
    CallToolResult,
    ErrorData,
    ImageContent,
    ListToolsResult,
    TextContent,
    Tool,
)

from flwr.supercore.typing import JSONObject

from .. import registry
from . import mcp as slack_mcp


def test_mcp_discovery_and_execution() -> None:
    """Discover all Slack tools and execute former static names through MCP."""
    session = create_autospec(ClientSession, instance=True)
    session.list_tools.side_effect = [
        ListToolsResult(
            tools=[
                Tool(name="slack_search_public", inputSchema={"type": "object"}),
                Tool(name="slack_read_channel", inputSchema={"type": "object"}),
            ],
            nextCursor="next",
        ),
        ListToolsResult(tools=[]),
    ]
    session.send_request.return_value = CallToolResult(
        content=[TextContent(type="text", text="done")]
    )
    credentials: JSONObject = {"access_token": "xoxp-secret"}
    assert not registry.get_connector_tools("slack")
    assert registry.is_dynamic_connector_tool("slack_search_public")
    with patch.object(slack_mcp, "_session") as open_session:
        open_session.return_value.__aenter__.return_value = session
        tools = registry.invoke_connector(
            "slack__discover_tools", {}, Mock(), credentials, {}
        )
        assert isinstance(tools, list)
        assert [cast(JSONObject, tool)["name"] for tool in tools] == [
            "slack_search_public",
            "slack_read_channel",
        ]
        result = registry.invoke_connector(
            "slack_search_public",
            {"query": "release"},
            Mock(),
            credentials,
            {},
        )
    assert session.list_tools.await_args_list == [
        call(cursor=None),
        call(cursor="next"),
    ]
    request = session.send_request.await_args.args[0].root
    assert request.params.name == "slack_search_public"
    assert request.params.arguments == {"query": "release"}
    assert isinstance(result, dict)
    assert result["content"] == [{"type": "text", "text": "done"}]


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (
            CallToolResult(
                isError=True,
                content=[
                    TextContent(type="text", text="Denied xoxp-secret"),
                    ImageContent(type="image", data="eA==", mimeType="image/png"),
                    TextContent(type="text", text="Missing scope: chat:write"),
                ],
            ),
            "Denied [REDACTED] Missing scope: chat:write",
        ),
        (CallToolResult(isError=True, content=[]), ""),
    ],
)
def test_mcp_tool_error_preserves_redacted_diagnostics(
    result: CallToolResult, message: str
) -> None:
    """Preserve error text without exposing the bound token or non-text blocks."""
    session = create_autospec(ClientSession, instance=True)
    session.send_request.return_value = result
    credentials: JSONObject = {"access_token": "xoxp-secret"}
    with patch.object(slack_mcp, "_session") as open_session:
        open_session.return_value.__aenter__.return_value = session
        with pytest.raises(slack_mcp.SlackApiError) as error:
            registry.invoke_connector(
                "slack_search_public", {"query": "release"}, Mock(), credentials, {}
            )
    assert error.value.code == "mcp_tool_error"
    assert error.value.message == message
    assert "xoxp-secret" not in "".join(traceback.format_exception(error.value))


@pytest.mark.parametrize("failure", ["http", "protocol", "connector"])
def test_mcp_session_error_preserves_redacted_diagnostics(failure: str) -> None:
    """Preserve nested session failures without exposing the bound token."""
    message = "Denied xoxp-secret"
    credentials: JSONObject = {"access_token": "xoxp-secret"}
    session = create_autospec(ClientSession, instance=True)
    session.__aenter__.return_value = session
    cause: Exception
    if failure == "http":
        cause = httpx.HTTPStatusError(
            message,
            request=httpx.Request("POST", "https://mcp.slack.com/mcp"),
            response=httpx.Response(401),
        )
    elif failure == "protocol":
        cause = McpError(ErrorData(code=-32602, message=message))
    else:
        cause = slack_mcp.SlackApiError("invalid_tools", message=message)
    session.list_tools.side_effect = ExceptionGroup(
        "MCP failed", [ExceptionGroup("Transport failed", [cause])]
    )
    with (
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.streamable_http.streamable_http_client") as transport,
    ):
        transport.return_value.__aenter__.return_value = (Mock(), Mock(), Mock())
        with pytest.raises(slack_mcp.SlackApiError) as error:
            registry.invoke_connector(
                "slack__discover_tools", {}, Mock(), credentials, {}
            )
    assert error.value.code == (
        "invalid_tools" if failure == "connector" else "request_failed"
    )
    assert error.value.status_code == (401 if failure == "http" else None)
    assert error.value.message == "Denied [REDACTED]"
    assert "xoxp-secret" not in "".join(traceback.format_exception(error.value))
