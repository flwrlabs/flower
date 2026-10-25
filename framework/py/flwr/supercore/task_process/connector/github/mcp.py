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
"""Discover and execute tools through GitHub's official MCP server."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import TYPE_CHECKING, cast

import httpx

from flwr.supercore.typing import JSONObject

from ..definition import ActionAccess, ActionDefinition, ConnectorExecutionContext
from ..http import ConnectorApiError

if TYPE_CHECKING:
    from mcp import ClientSession

_MCP_URL = "https://api.githubcopilot.com/mcp/"


class GitHubApiError(ConnectorApiError):
    """GitHub failure forwarded through the connector error channel."""

    provider = "GitHub"


@asynccontextmanager
async def _session(context: ConnectorExecutionContext) -> AsyncIterator[ClientSession]:
    """Open an authenticated MCP session using this task's bound credentials."""
    # pylint: disable=import-outside-toplevel
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ModuleNotFoundError:
        raise ImportError(
            "GitHub MCP requires the optional 'flwr[agent]' dependencies."
        ) from None

    token = context.credentials.get("access_token")
    if not isinstance(token, str) or not token:
        raise GitHubApiError("invalid_credentials")
    try:
        async with (
            asyncio.timeout(120),
            httpx.AsyncClient(
                headers={"Authorization": f"Bearer {token}", "X-MCP-Toolsets": "all"},
                timeout=30,
                follow_redirects=False,
            ) as client,
            streamable_http_client(_MCP_URL, http_client=client) as (read, write, _),
        ):
            async with ClientSession(
                read, write, read_timeout_seconds=timedelta(seconds=30)
            ) as session:
                await session.initialize()
                yield session
        return
    except Exception as ex:  # pylint: disable=broad-exception-caught
        while isinstance(ex, ExceptionGroup):
            ex = ex.exceptions[0]  # pylint: disable=no-member
        if isinstance(ex, GitHubApiError):
            code, status, message = ex.code, ex.status_code, ex.message
        else:
            code = "request_failed"
            status = (
                ex.response.status_code
                if isinstance(ex, httpx.HTTPStatusError)
                else None
            )
            message = str(ex)
        error = GitHubApiError(
            code, status, (message or "").replace(token, "[REDACTED]")
        )
    # Suppress the original exception so tracebacks cannot expose the token.
    raise error from None


def discover_tools(context: ConnectorExecutionContext) -> list[ActionDefinition]:
    """Discover account-specific tools from GitHub's MCP server."""
    return asyncio.run(_discover_tools(context))


async def _discover_tools(context: ConnectorExecutionContext) -> list[ActionDefinition]:
    """Follow tools/list cursors and preserve GitHub's remote schemas and names."""
    actions: list[ActionDefinition] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    async with _session(context) as session:
        while True:
            page = await session.list_tools(cursor=cursor)
            for tool in page.tools:
                actions.append(
                    ActionDefinition(
                        name=tool.name,
                        description=tool.description or "",
                        access=(
                            ActionAccess.READ
                            if tool.annotations and tool.annotations.readOnlyHint
                            else ActionAccess.WRITE
                        ),
                        input_schema=cast(JSONObject, tool.inputSchema),
                    )
                )
            cursor = page.nextCursor
            if not cursor:
                return actions
            if cursor in seen_cursors:
                raise GitHubApiError("invalid_tools")
            seen_cursors.add(cursor)


def execute_tool(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Execute a discovered action without fetching the catalog again."""
    return asyncio.run(_execute_tool(arguments, context))


async def _execute_tool(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Forward the original remote name and retain the MCP result content."""
    if not context.action_name:
        raise GitHubApiError("invalid_tool")
    async with _session(context) as session:
        # pylint: disable-next=import-outside-toplevel
        from mcp import types

        # call_tool also lists tools to validate output schemas in a fresh session.
        result = await session.send_request(
            types.ClientRequest(
                types.CallToolRequest(
                    params=types.CallToolRequestParams(
                        name=context.action_name, arguments=arguments
                    )
                )
            ),
            types.CallToolResult,
        )
        if result.isError:
            raise GitHubApiError(
                "tool_error",
                message=" ".join(
                    item.text
                    for item in result.content
                    if isinstance(item, types.TextContent)
                ),
            )
    return cast(
        JSONObject, result.model_dump(mode="json", by_alias=True, exclude_none=True)
    )
