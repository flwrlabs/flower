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
"""Shared client for Google's remote Workspace MCP servers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, cast

import httpx

from flwr.supercore.typing import JSONObject

from ..definition import ActionAccess, ActionDefinition, ConnectorExecutionContext
from ..http import ConnectorApiError

if TYPE_CHECKING:
    from mcp import ClientSession


class GoogleMcpApiError(ConnectorApiError):
    """Secret-safe Google Workspace MCP failure."""

    provider = "Google Workspace"


@dataclass(frozen=True)
class GoogleMcpClient:
    """Discover and execute tools on one authenticated Google MCP endpoint."""

    url: str

    @asynccontextmanager
    async def _session(
        self, context: ConnectorExecutionContext
    ) -> AsyncIterator[ClientSession]:
        """Open a session with the access token supplied by SuperLink."""
        # pylint: disable=import-outside-toplevel
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
        except ModuleNotFoundError:
            raise ImportError(
                "Google MCP requires the optional 'flwr[agent]' dependencies."
            ) from None

        token = context.credentials.get("access_token")
        if not isinstance(token, str) or not token:
            raise GoogleMcpApiError("invalid_credentials")
        try:
            async with httpx.AsyncClient(
                headers={"Authorization": f"Bearer {token}"},
                timeout=httpx.Timeout(30.0, read=60.0),
            ) as client:
                async with streamable_http_client(self.url, http_client=client) as (
                    read,
                    write,
                    _,
                ):
                    async with ClientSession(
                        read, write, read_timeout_seconds=timedelta(seconds=60)
                    ) as session:
                        await session.initialize()
                        yield session
        except GoogleMcpApiError:
            raise
        except Exception:  # pylint: disable=broad-exception-caught
            raise GoogleMcpApiError("mcp_error") from None

    def discover_tools(
        self, context: ConnectorExecutionContext
    ) -> list[ActionDefinition]:
        """Discover all tools available to the bound Google account."""
        return asyncio.run(self._discover_tools(context))

    async def _discover_tools(
        self, context: ConnectorExecutionContext
    ) -> list[ActionDefinition]:
        """Follow tools/list cursors while preserving native Google tool names."""
        actions: list[ActionDefinition] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        async with self._session(context) as session:
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
                    raise GoogleMcpApiError("invalid_tools")
                seen_cursors.add(cursor)

    def execute_tool(
        self, arguments: JSONObject, context: ConnectorExecutionContext
    ) -> JSONObject:
        """Execute a native Google tool without fetching the catalog again."""
        return asyncio.run(self._execute_tool(arguments, context))

    async def _execute_tool(
        self, arguments: JSONObject, context: ConnectorExecutionContext
    ) -> JSONObject:
        """Preserve structured and multimodal results, and redact error tokens."""
        if not context.action_name:
            raise GoogleMcpApiError("invalid_tool")
        async with self._session(context) as session:
            # pylint: disable-next=import-outside-toplevel
            from mcp import types

            # call_tool also lists tools to validate schemas in a fresh session.
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
            message = " ".join(
                item.text
                for item in result.content
                if isinstance(item, types.TextContent)
            )
            for secret in (
                context.credentials.get("access_token"),
                context.credentials.get("refresh_token"),
            ):
                if isinstance(secret, str) and secret:
                    message = message.replace(secret, "[REDACTED]")
            raise GoogleMcpApiError("mcp_tool_error", message=message)
        return cast(
            JSONObject, result.model_dump(mode="json", by_alias=True, exclude_none=True)
        )
