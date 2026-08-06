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
"""Read-only Notion connector tools."""

from collections.abc import Callable

import requests

from flwr.supercore.task_process.usage import TaskUsageRecorder
from flwr.supercore.typing import JSONObject, JSONValue

from .http import ConnectorApiError, request_json_object
from .json_utils import optional_string, require_int_range, require_string
from .tool_schema import function_tool, integer_property, string_property

NOTION_CONNECTOR_REF = "notion"
NOTION_SEARCH_TOOL = "notion_search"
NOTION_GET_PAGE_CONTENT_TOOL = "notion_get_page_content"
NOTION_API_VERSION = "2026-03-11"

NOTION_TOOL_NAMES = (NOTION_SEARCH_TOOL, NOTION_GET_PAGE_CONTENT_TOOL)
NOTION_TOOLS = (
    function_tool(
        NOTION_SEARCH_TOOL,
        "Search pages and data sources shared with Notion.",
        properties={
            "query": string_property("Text contained in the Notion title."),
            "limit": integer_property(
                "Maximum number of results to return.", minimum=1, maximum=100
            ),
            "cursor": string_property("Cursor returned by the previous response."),
        },
        required=("query",),
    ),
    function_tool(
        NOTION_GET_PAGE_CONTENT_TOOL,
        "Read one page of a Notion page's block content.",
        properties={
            "page_id": string_property("Notion page ID returned by search."),
            "max_blocks": integer_property(
                "Maximum number of blocks to return.", minimum=1, maximum=100
            ),
            "cursor": string_property("Cursor returned by the previous response."),
        },
        required=("page_id",),
    ),
)

_NOTION_API_BASE_URL = "https://api.notion.com/v1"


class NotionApiError(ConnectorApiError):
    """Secret-safe Notion API failure."""

    provider = "Notion"


# pylint: disable-next=too-many-arguments
def search(
    query: str,
    limit: int = 10,
    cursor: str | None = None,
    *,
    credentials: JSONObject,
    config: JSONObject,
    usage_recorder: TaskUsageRecorder,
) -> JSONObject:
    """Search pages and data sources shared with the Notion connection."""
    del config, usage_recorder
    body: JSONObject = {
        "query": require_string(query, "Notion", "query"),
        "page_size": require_int_range(limit, "Notion", "limit", maximum=100),
    }
    if cursor := optional_string(cursor, "Notion", "cursor"):
        body["start_cursor"] = cursor
    return _call_notion_api("POST", "/search", credentials, body=body)


# pylint: disable-next=too-many-arguments
def get_page_content(
    page_id: str,
    max_blocks: int = 100,
    cursor: str | None = None,
    *,
    credentials: JSONObject,
    config: JSONObject,
    usage_recorder: TaskUsageRecorder,
) -> JSONObject:
    """Read one page of a Notion page's block content."""
    del config, usage_recorder
    params = {
        "page_size": str(
            require_int_range(max_blocks, "Notion", "max_blocks", maximum=100)
        )
    }
    if cursor := optional_string(cursor, "Notion", "cursor"):
        params["start_cursor"] = cursor
    page_id = require_string(page_id, "Notion", "page_id")
    return _call_notion_api(
        "GET", f"/blocks/{page_id}/children", credentials, params=params
    )


NOTION_TOOL_HANDLERS: dict[str, Callable[..., JSONValue]] = {
    NOTION_SEARCH_TOOL: search,
    NOTION_GET_PAGE_CONTENT_TOOL: get_page_content,
}


def _call_notion_api(
    method: str,
    path: str,
    credentials: JSONObject,
    *,
    body: JSONObject | None = None,
    params: dict[str, str] | None = None,
) -> JSONObject:
    """Call one Notion API endpoint and return its JSON response."""
    token = credentials.get("access_token")
    if not isinstance(token, str) or not token:
        raise NotionApiError("invalid_credentials")
    return request_json_object(
        method,
        f"{_NOTION_API_BASE_URL}{path}",
        error=NotionApiError,
        headers={
            "Authorization": f"Bearer {token}",
            "Notion-Version": NOTION_API_VERSION,
        },
        params=params,
        json=body,
        http_error_code=_response_error_code,
    )


def _response_error_code(response: requests.Response) -> str:
    """Return a documented Notion error code without response details."""
    if response.status_code == 429:
        return "rate_limited"
    try:
        code = response.json().get("code")
    except (AttributeError, ValueError):
        return "http_error"
    if isinstance(code, str) and code.replace("_", "").isalnum() and code.islower():
        return code
    return "http_error"
