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

import asyncio
from datetime import timedelta
from typing import cast

import httpx

from flwr.supercore.typing import JSONObject, JSONValue

from ..http import ConnectorApiError

_URL = "https://api.githubcopilot.com/mcp/"


class GitHubApiError(ConnectorApiError):
    """GitHub failure forwarded through the connector error channel."""

    provider = "GitHub"


def request(
    name: str | None, arguments: JSONObject, credentials: JSONObject
) -> JSONValue:
    """Discover tools or call one, forwarding errors without exposing the token."""
    token = credentials.get("access_token")
    if not isinstance(token, str) or not token:
        raise GitHubApiError("invalid_credentials")
    try:
        return asyncio.run(_request(name, arguments, token))
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
    # Raise outside the handler so the original exception cannot expose the token.
    raise error


async def _request(name: str | None, arguments: JSONObject, token: str) -> JSONValue:
    """Initialize a task-local MCP session and discover or call remote tools."""
    from mcp import ClientSession, types  # pylint: disable=import-outside-toplevel
    from mcp.client.streamable_http import (  # pylint: disable=import-outside-toplevel
        streamable_http_client,
    )

    async with (
        asyncio.timeout(120),
        httpx.AsyncClient(
            headers={"Authorization": f"Bearer {token}", "X-MCP-Toolsets": "all"},
            timeout=30,
            follow_redirects=False,
        ) as client,
        streamable_http_client(_URL, http_client=client) as (read, write, _),
    ):
        async with ClientSession(
            read, write, read_timeout_seconds=timedelta(seconds=30)
        ) as session:
            await session.initialize()
            if name is None:
                tools: list[JSONObject] = []
                params: types.PaginatedRequestParams | None = None
                while True:
                    page = await session.list_tools(params=params)
                    tools.extend(
                        cast(
                            JSONObject,
                            tool.model_dump(
                                mode="json", by_alias=True, exclude_none=True
                            ),
                        )
                        for tool in page.tools
                    )
                    if not page.nextCursor:
                        return cast(JSONValue, tools)
                    params = types.PaginatedRequestParams(cursor=page.nextCursor)
            result = await session.call_tool(name, arguments)
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
                JSONObject,
                result.model_dump(mode="json", by_alias=True, exclude_none=True),
            )
