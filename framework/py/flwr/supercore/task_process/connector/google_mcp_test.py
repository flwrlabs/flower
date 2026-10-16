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
"""Smoke tests for Google Workspace MCP connectors."""

from typing import cast
from unittest.mock import Mock, create_autospec, patch

import pytest
from mcp import ClientSession
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from flwr.supercore.typing import JSONObject

from . import registry
from .google_mcp import GoogleMcpApiError, GoogleMcpClient

_CREDENTIALS: JSONObject = {
    "access_token": "access-token",
    "refresh_token": "refresh-secret",
}


@pytest.mark.parametrize(
    ("connector", "action"),
    [
        ("google_drive", "search_files"),
        ("gmail", "search_threads"),
        ("google_calendar", "list_events"),
    ],
)
def test_google_discovery_and_execution(connector: str, action: str) -> None:
    """Expose prefixed tools and forward native names and arguments to MCP."""
    session = create_autospec(ClientSession, instance=True)
    session.list_tools.return_value = ListToolsResult(
        tools=[Tool(name=action, inputSchema={"type": "object"})]
    )
    session.send_request.return_value = CallToolResult(
        content=[], structuredContent={"items": []}
    )
    with patch.object(GoogleMcpClient, "_session") as open_session:
        open_session.return_value.__aenter__.return_value = session
        tools = registry.invoke_connector(
            f"{connector}__discover_tools", {}, Mock(), _CREDENTIALS, {}
        )
        assert isinstance(tools, list)
        assert [cast(JSONObject, tool)["name"] for tool in tools] == [
            f"{connector}_{action}"
        ]
        result = registry.invoke_connector(
            f"{connector}_{action}", {"pageSize": 7}, Mock(), _CREDENTIALS, {}
        )
    request = session.send_request.await_args.args[0].root
    assert request.params.name == action
    assert request.params.arguments == {"pageSize": 7}
    assert isinstance(result, dict)
    assert result["structuredContent"] == {"items": []}
    session.list_tools.assert_awaited_once()


def test_google_redacts_error_credentials() -> None:
    """Hide credentials while preserving provider diagnostics."""
    session = create_autospec(ClientSession, instance=True)
    session.send_request.return_value = CallToolResult(
        isError=True,
        content=[TextContent(type="text", text="Denied access-token refresh-secret")],
    )
    with patch.object(GoogleMcpClient, "_session") as open_session:
        open_session.return_value.__aenter__.return_value = session
        with pytest.raises(GoogleMcpApiError) as error:
            registry.invoke_connector(
                "google_drive_search_files", {}, Mock(), _CREDENTIALS, {}
            )
    assert error.value.code == "mcp_tool_error"
    assert error.value.message == "Denied [REDACTED] [REDACTED]"
