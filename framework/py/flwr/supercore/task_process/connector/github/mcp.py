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
"""Discover and call GitHub's official remote MCP tools."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import TYPE_CHECKING, cast

import httpx

from flwr.supercore.typing import JSONObject, JSONValue

from ..http import ConnectorApiError

_URL = "https://api.githubcopilot.com/mcp/"
_REQUEST_TIMEOUT = 30.0
_CALL_TIMEOUT = 120.0

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp.types import Tool


class GitHubApiError(ConnectorApiError):
    """Secret-safe GitHub API failure."""

    provider = "GitHub"


def call_tool(name: str, arguments: JSONObject, credentials: JSONObject) -> JSONObject:
    """Call any tool advertised for the connection's saved GitHub token."""
    return cast(JSONObject, _run(name, arguments, credentials))


def discover_tools(credentials: JSONObject) -> list[JSONObject]:
    """Return model-facing schemas for all tools offered by GitHub."""
    return cast(list[JSONObject], _run(None, {}, credentials))


def _run(name: str | None, arguments: JSONObject, credentials: JSONObject) -> JSONValue:
    """Run discovery or invocation without retaining credential-bearing errors."""
    token = credentials.get("access_token")
    if not isinstance(token, str) or not token:
        raise GitHubApiError("invalid_credentials")
    try:
        return asyncio.run(_call_tool(name, arguments, token))
    except Exception as ex:  # pylint: disable=broad-exception-caught
        error = _safe_error(ex)
    # Do not retain a transport exception (which may contain credentials) as context.
    raise error


async def _call_tool(name: str | None, arguments: JSONObject, token: str) -> JSONValue:
    """Initialize, discover and invoke a tool in a task-local MCP session."""
    # Control-plane services load the registry without executing connector tools.
    try:
        from mcp import ClientSession, types  # pylint: disable=import-outside-toplevel
        from mcp.client.streamable_http import (  # pylint: disable=import-outside-toplevel
            streamable_http_client,
        )
    except ImportError:
        raise GitHubApiError(
            "missing_dependency", message="Install the MCP SDK in the connector runtime"
        ) from None

    async with (
        asyncio.timeout(_CALL_TIMEOUT),
        httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {token}",
                "X-MCP-Toolsets": "all",
            },
            timeout=_REQUEST_TIMEOUT,
            follow_redirects=False,
        ) as client,
    ):
        async with streamable_http_client(_URL, http_client=client) as (read, write, _):
            async with ClientSession(
                read, write, read_timeout_seconds=timedelta(seconds=_REQUEST_TIMEOUT)
            ) as session:
                await session.initialize()
                tools = await _list_tools(session)
                if name is None:
                    return [_model_tool(tool) for tool in tools]
                remote_tool = next(
                    (tool for tool in tools if tool.name.lower() == name), None
                )
                if remote_tool is None:
                    raise GitHubApiError(
                        "tool_unavailable",
                        message="Check GitHub connection permissions",
                    )
                result = await session.call_tool(remote_tool.name, arguments)
                if result.isError:
                    message = " ".join(
                        item.text
                        for item in result.content
                        if isinstance(item, types.TextContent)
                    )
                    raise GitHubApiError(
                        "tool_error",
                        message=message.replace(token, "[REDACTED]") or None,
                    )
                return cast(
                    JSONObject,
                    result.model_dump(mode="json", by_alias=True, exclude_none=True),
                )


async def _list_tools(session: ClientSession) -> list[Tool]:
    """Read every catalog page and reject ambiguous names or cursor cycles."""
    from mcp.types import (  # pylint: disable=import-outside-toplevel
        PaginatedRequestParams,
    )

    tools: list[Tool] = []
    names: set[str] = set()
    cursors: set[str] = set()
    cursor: str | None = None
    while True:
        page = await session.list_tools(
            params=PaginatedRequestParams(cursor=cursor) if cursor is not None else None
        )
        for tool in page.tools:
            name = tool.name.lower()
            if name in names:
                raise GitHubApiError("invalid_response")
            names.add(name)
            tools.append(tool)
        cursor = page.nextCursor
        if not cursor:
            return tools
        if cursor in cursors:
            raise GitHubApiError("invalid_response")
        cursors.add(cursor)


def _model_tool(tool: Tool) -> JSONObject:
    """Namespace a remote tool while preserving its description and input schema."""
    return {
        "type": "function",
        "name": f"github_{tool.name.lower()}",
        "description": tool.description or "",
        "parameters": cast(JSONObject, tool.inputSchema),
        "strict": False,
    }


def _safe_error(error: Exception) -> GitHubApiError:
    """Keep safe connector errors and discard transport diagnostic text."""
    if isinstance(error, ExceptionGroup):
        return _safe_error(error.exceptions[0])
    if isinstance(error, GitHubApiError):
        return GitHubApiError(error.code, error.status_code, error.message)
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        message = (
            "Reconnect GitHub and check the connection's repository permissions"
            if status in (401, 403)
            else None
        )
        return GitHubApiError("http_error", status, message)
    if isinstance(error, (TimeoutError, httpx.TimeoutException)):
        return GitHubApiError("timeout")
    return GitHubApiError("request_failed")
