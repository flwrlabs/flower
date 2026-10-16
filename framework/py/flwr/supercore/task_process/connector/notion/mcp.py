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
"""Discover and execute tools through Notion's official MCP server."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import timedelta
from typing import TYPE_CHECKING, cast

import httpx

from flwr.supercore.typing import JSONObject

from ..definition import ActionAccess, ActionDefinition, ConnectorExecutionContext
from ..http import ConnectorApiError

if TYPE_CHECKING:
    from mcp import ClientSession

_MCP_URL = "https://mcp.notion.com/mcp"
_SSE_URL = "https://mcp.notion.com/sse"


class NotionApiError(ConnectorApiError):
    """Notion failure forwarded through the connector error channel."""

    provider = "Notion"


@asynccontextmanager
async def _session(context: ConnectorExecutionContext) -> AsyncIterator[ClientSession]:
    """Try Streamable HTTP, falling back to SSE before handing out the session."""
    token = context.credentials.get("access_token")
    if not isinstance(token, str) or not token:
        raise NotionApiError("invalid_credentials")
    try:
        async with asyncio.timeout(120), AsyncExitStack() as stack:
            try:
                session = await stack.enter_async_context(
                    _initialized_session(token, use_sse=False)
                )
            except Exception:  # pylint: disable=broad-exception-caught
                session = await stack.enter_async_context(
                    _initialized_session(token, use_sse=True)
                )
            yield session
        return
    except Exception as ex:  # pylint: disable=broad-exception-caught
        while isinstance(ex, ExceptionGroup):
            ex = ex.exceptions[0]  # pylint: disable=no-member
        if isinstance(ex, NotionApiError):
            code, status, message = ex.code, ex.status_code, ex.message
        else:
            code = "request_failed"
            status = (
                ex.response.status_code
                if isinstance(ex, httpx.HTTPStatusError)
                else None
            )
            message = str(ex)
        error = NotionApiError(
            code, status, (message or "").replace(token, "[REDACTED]")
        )
    # Suppress the original exception so tracebacks cannot expose the token.
    raise error from None


@asynccontextmanager
async def _initialized_session(
    token: str, *, use_sse: bool
) -> AsyncIterator[ClientSession]:
    """Open one transport and clean up if its connection or initialization fails."""
    # pylint: disable=import-outside-toplevel
    try:
        from mcp import ClientSession
        from mcp.client.sse import sse_client
        from mcp.client.streamable_http import streamable_http_client
    except ModuleNotFoundError:
        raise ImportError(
            "Notion MCP requires the optional 'flwr[agent]' dependencies."
        ) from None

    headers = {"Authorization": f"Bearer {token}"}
    async with AsyncExitStack() as stack:
        if use_sse:
            read, write = await stack.enter_async_context(
                sse_client(_SSE_URL, headers=headers, timeout=30, sse_read_timeout=30)
            )
        else:
            client = await stack.enter_async_context(
                httpx.AsyncClient(headers=headers, timeout=30, follow_redirects=False)
            )
            read, write, _ = await stack.enter_async_context(
                streamable_http_client(_MCP_URL, http_client=client)
            )
        session = await stack.enter_async_context(
            ClientSession(read, write, read_timeout_seconds=timedelta(seconds=30))
        )
        await session.initialize()
        yield session


def discover_tools(context: ConnectorExecutionContext) -> list[ActionDefinition]:
    """Discover account-specific tools from Notion's MCP server."""
    return asyncio.run(_discover_tools(context))


async def _discover_tools(context: ConnectorExecutionContext) -> list[ActionDefinition]:
    """Follow tools/list cursors and preserve Notion's remote schemas and names."""
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
                raise NotionApiError("invalid_tools")
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
        raise NotionApiError("invalid_tool")
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
            raise NotionApiError(
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
