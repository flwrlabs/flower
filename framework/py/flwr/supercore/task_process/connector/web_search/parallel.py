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
"""Anonymous Parallel Unified MCP web search adapter."""

import json
from collections.abc import Iterator, Sequence
from typing import cast

import requests

from flwr.supercore.typing import JSONObject, JSONValue
from flwr.supercore.version import package_name, package_version

PARALLEL_MCP_URL = "https://mcp.parallel.ai/mcp"
REQUEST_TIMEOUT = 60.0


class ParallelWebSearchProvider:
    """Search with Parallel's anonymous, rate-limited MCP service."""

    name = "parallel"
    env = ""  # This last fallback does not require a provider environment variable.

    def search(self, query: str) -> JSONObject:
        """Execute one search using the server's default Fast mode."""
        if not query.strip():
            raise ValueError("web search requires a non-empty query.")
        query = query.strip()
        with requests.Session() as session:
            session.headers.update(
                {
                    "User-Agent": f"{package_name}/{package_version}",
                    "Accept": "application/json, text/event-stream",
                    "Content-Type": "application/json",
                }
            )
            try:
                initialized = _rpc(
                    session,
                    "initialize",
                    1,
                    {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {
                            "name": package_name,
                            "version": package_version,
                        },
                    },
                )
                version = initialized.get("protocolVersion")
                if not isinstance(version, str):
                    raise RuntimeError("parallel MCP returned no protocol version.")
                session.headers["MCP-Protocol-Version"] = version
                _rpc(session, "notifications/initialized")
                tools = _rpc(session, "tools/list", 2).get("tools")
                if not isinstance(tools, list) or not any(
                    isinstance(tool, dict) and tool.get("name") == "web_search"
                    for tool in tools
                ):
                    raise RuntimeError("parallel MCP has no web_search tool.")
                result = _rpc(
                    session,
                    "tools/call",
                    3,
                    {
                        "name": "web_search",
                        "arguments": {"objective": query, "search_queries": [query]},
                    },
                )
                return {
                    "results": cast(
                        Sequence[JSONValue], _parse_results(_search_content(result))
                    )
                }
            except requests.RequestException as exc:
                raise RuntimeError("parallel web search request failed.") from exc
            except ValueError as exc:
                raise RuntimeError("parallel MCP returned invalid JSON.") from exc
            finally:
                if "Mcp-Session-Id" in session.headers:
                    # Session termination is optional; do not mask the search outcome.
                    try:
                        session.delete(
                            PARALLEL_MCP_URL,
                            timeout=REQUEST_TIMEOUT,
                            allow_redirects=False,
                            stream=True,
                        ).close()
                    except requests.RequestException:
                        pass


def _rpc(
    session: requests.Session,
    method: str,
    request_id: int | None = None,
    params: JSONObject | None = None,
) -> JSONObject:
    body: JSONObject = {"jsonrpc": "2.0", "method": method}
    if request_id is not None:
        body["id"] = request_id
    if params is not None:
        body["params"] = params
    with session.post(
        PARALLEL_MCP_URL,
        json=body,
        timeout=REQUEST_TIMEOUT,
        allow_redirects=False,
        stream=True,
    ) as response:
        if not 200 <= response.status_code < 300:
            raise RuntimeError(
                f"parallel MCP request failed: HTTP {response.status_code}."
            )
        if session_id := response.headers.get("Mcp-Session-Id"):
            session.headers["Mcp-Session-Id"] = session_id
        if request_id is None:
            return {}
        messages = _read_messages(response)
        for message in messages:
            if not isinstance(message, dict) or message.get("id") != request_id:
                continue
            if "error" in message:
                raise RuntimeError("parallel MCP returned an RPC error.")
            result = message.get("result")
            if isinstance(result, dict):
                return result
            break
        raise RuntimeError("parallel MCP returned no matching result.")


def _read_messages(response: requests.Response) -> Iterator[JSONValue]:
    media_type = (
        response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    )
    if media_type != "text/event-stream":
        yield cast(JSONValue, response.json())
        return
    response.encoding = "utf-8"
    data: list[str] = []
    line = ""
    skip_lf = False
    # A small reply can arrive on a stream that stays open. Read without waiting
    # for a full chunk, using requests' incremental UTF-8 decoder.
    for char in response.iter_content(chunk_size=1, decode_unicode=True):
        if skip_lf and char == "\n":
            skip_lf = False
            continue
        skip_lf = char == "\r"
        if char not in ("\r", "\n"):
            line += char
            continue
        if not line and data:
            yield json.loads("\n".join(data))
            data.clear()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
        line = ""
    if line.startswith("data:"):
        data.append(line[5:].lstrip())
    if data:
        yield json.loads("\n".join(data))


def _search_content(result: JSONObject) -> JSONObject:
    if result.get("isError"):
        raise RuntimeError("parallel MCP web_search tool failed.")
    payload = result.get("structuredContent")
    if isinstance(payload, dict):
        return payload
    # MCP also permits JSON encoded in a text content block.
    content = result.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    payload = json.loads(text)
                    if isinstance(payload, dict):
                        return payload
    raise RuntimeError("parallel MCP returned invalid search content.")


def _parse_results(payload: JSONObject) -> list[JSONObject]:
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise RuntimeError("parallel MCP response must include a results list.")
    results: list[JSONObject] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        title, url = item.get("title"), item.get("url")
        if not isinstance(url, str) or not url.strip():
            continue
        excerpts = item.get("excerpts")
        snippet = (
            "\n\n".join(
                excerpt.strip()
                for excerpt in excerpts
                if isinstance(excerpt, str) and excerpt.strip()
            )
            if isinstance(excerpts, list)
            else ""
        )
        published_at = item.get("publish_date")
        results.append(
            {
                "title": (
                    title.strip()
                    if isinstance(title, str) and title.strip()
                    else url.strip()
                ),
                "url": url.strip(),
                "snippet": snippet or None,
                "published_at": (
                    published_at.strip()
                    if isinstance(published_at, str) and published_at.strip()
                    else None
                ),
            }
        )
    return results
