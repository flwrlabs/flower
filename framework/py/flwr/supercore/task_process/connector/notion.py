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

import re
from collections.abc import Callable
from typing import cast

import requests

from flwr.supercore.task_process.usage import TaskUsageRecorder
from flwr.supercore.typing import JSONObject, JSONValue

NOTION_CONNECTOR_REF = "notion"
NOTION_SEARCH_TOOL = "notion_search"
NOTION_GET_PAGE_CONTENT_TOOL = "notion_get_page_content"
NOTION_API_VERSION = "2026-03-11"

NOTION_TOOL_NAMES = (
    NOTION_SEARCH_TOOL,
    NOTION_GET_PAGE_CONTENT_TOOL,
)

_NOTION_API_BASE_URL = "https://api.notion.com/v1"
_REQUEST_TIMEOUT = 30.0
_SAFE_ERROR_CODE = re.compile(r"^[a-z0-9_]+$")


class NotionApiError(RuntimeError):
    """Secret-safe Notion API failure."""

    def __init__(self, code: str, status_code: int | None = None) -> None:
        self.code = code
        self.status_code = status_code
        detail = code if status_code is None else f"{code} ({status_code})"
        super().__init__(f"Notion API request failed: {detail}.")


def make_notion_tools() -> list[JSONObject]:
    """Return model-facing schemas for Notion's read-only v1 operations."""
    return [
        {
            "type": "function",
            "name": NOTION_SEARCH_TOOL,
            "description": "Search pages and data sources shared with Notion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text contained in the Notion title.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Maximum number of results to return.",
                    },
                    "cursor": {
                        "type": "string",
                        "description": "Cursor returned by the previous response.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": NOTION_GET_PAGE_CONTENT_TOOL,
            "description": (
                "Read a Notion page's blocks, including nested block content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "Notion page ID returned by search.",
                    },
                    "max_blocks": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 200,
                        "description": "Maximum number of blocks to return.",
                    },
                },
                "required": ["page_id"],
                "additionalProperties": False,
            },
        },
    ]


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
    limit = _bounded_int(limit, "limit", maximum=100)
    body: JSONObject = {
        "query": _non_empty_string(query, "query"),
        "page_size": limit,
    }
    cursor = _optional_cursor(cursor)
    if cursor is not None:
        body["start_cursor"] = cursor
    payload = _call_notion_api("POST", "/search", credentials, body=body)
    results = _required_object_list(payload, "results")
    has_more, next_cursor = _pagination(payload)
    return {
        "results": [_normalize_search_result(item) for item in results[:limit]],
        "has_more": has_more,
        "next_cursor": next_cursor,
    }


def get_page_content(
    page_id: str,
    max_blocks: int = 100,
    *,
    credentials: JSONObject,
    config: JSONObject,
    usage_recorder: TaskUsageRecorder,
) -> JSONObject:
    """Read a Notion page's block content in depth-first document order."""
    del config, usage_recorder
    page_id = _non_empty_string(page_id, "page_id")
    max_blocks = _bounded_int(max_blocks, "max_blocks", maximum=200)
    blocks: list[JSONObject] = []
    truncated = _collect_block_children(
        page_id,
        credentials=credentials,
        blocks=blocks,
        max_blocks=max_blocks,
        depth=0,
    )
    return {
        "page_id": page_id,
        "blocks": blocks,
        "truncated": truncated,
    }


NOTION_TOOL_HANDLERS: dict[str, Callable[..., JSONValue]] = {
    NOTION_SEARCH_TOOL: search,
    NOTION_GET_PAGE_CONTENT_TOOL: get_page_content,
}


def _collect_block_children(
    block_id: str,
    *,
    credentials: JSONObject,
    blocks: list[JSONObject],
    max_blocks: int,
    depth: int,
) -> bool:
    """Append nested block children until complete or the limit is reached."""
    cursor: str | None = None
    while True:
        remaining = max_blocks - len(blocks)
        if remaining == 0:
            return True
        params = {"page_size": str(min(remaining, 100))}
        if cursor is not None:
            params["start_cursor"] = cursor
        payload = _call_notion_api(
            "GET",
            f"/blocks/{block_id}/children",
            credentials,
            params=params,
        )
        children = _required_object_list(payload, "results")
        for child in children:
            if len(blocks) == max_blocks:
                return True
            blocks.append(_normalize_block(child, depth=depth))
            if child.get("has_children") is True:
                if len(blocks) == max_blocks:
                    return True
                child_id = _required_string(child, "id")
                if _collect_block_children(
                    child_id,
                    credentials=credentials,
                    blocks=blocks,
                    max_blocks=max_blocks,
                    depth=depth + 1,
                ):
                    return True
        has_more, cursor = _pagination(payload)
        if not has_more:
            return False


def _call_notion_api(
    method: str,
    path: str,
    credentials: JSONObject,
    *,
    body: JSONObject | None = None,
    params: dict[str, str] | None = None,
) -> JSONObject:
    """Call one Notion API endpoint and validate its response envelope."""
    access_token = credentials.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise NotionApiError("invalid_credentials")
    try:
        response = requests.request(
            method,
            f"{_NOTION_API_BASE_URL}{path}",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Notion-Version": NOTION_API_VERSION,
            },
            json=body,
            params=params,
            timeout=_REQUEST_TIMEOUT,
        )
    except requests.RequestException:
        raise NotionApiError("request_failed") from None
    if response.status_code >= 400:
        code = _response_error_code(response)
        raise NotionApiError(code, status_code=response.status_code)
    try:
        payload = response.json()
    except ValueError:
        raise NotionApiError("invalid_response") from None
    if not isinstance(payload, dict):
        raise NotionApiError("invalid_response")
    return cast(JSONObject, payload)


def _response_error_code(response: requests.Response) -> str:
    """Return a documented Notion error code without exposing response text."""
    if response.status_code == 429:
        return "rate_limited"
    try:
        payload = response.json()
    except ValueError:
        return "http_error"
    if isinstance(payload, dict):
        code = payload.get("code")
        if isinstance(code, str) and _SAFE_ERROR_CODE.fullmatch(code):
            return code
    return "http_error"


def _normalize_search_result(item: JSONObject) -> JSONObject:
    """Return the stable subset of a Notion search result."""
    return {
        "id": _string_field(item, "id"),
        "object": _string_field(item, "object"),
        "title": _search_result_title(item),
        "url": _string_field(item, "url"),
        "last_edited_time": _string_field(item, "last_edited_time"),
    }


def _search_result_title(item: JSONObject) -> str:
    """Extract a page or data source title from a search result."""
    properties = item.get("properties")
    if isinstance(properties, dict):
        for prop in properties.values():
            if isinstance(prop, dict) and prop.get("type") == "title":
                return _rich_text(prop.get("title"))
    title = _rich_text(item.get("title"))
    if title:
        return title
    return _string_field(item, "name")


def _normalize_block(block: JSONObject, *, depth: int) -> JSONObject:
    """Return a compact model-facing representation of one Notion block."""
    block_type = _string_field(block, "type")
    block_data = block.get(block_type)
    return {
        "id": _string_field(block, "id"),
        "type": block_type,
        "text": _block_text(block_data),
        "depth": depth,
        "has_children": block.get("has_children") is True,
    }


def _block_text(block_data: JSONValue | None) -> str:
    """Extract readable text from common Notion block payloads."""
    if not isinstance(block_data, dict):
        return ""
    text = _rich_text(block_data.get("rich_text"))
    if text:
        return text
    title = block_data.get("title")
    if isinstance(title, str):
        return title
    cells = block_data.get("cells")
    if isinstance(cells, list):
        return " | ".join(_rich_text(cell) for cell in cells)
    caption = _rich_text(block_data.get("caption"))
    if caption:
        return caption
    expression = block_data.get("expression")
    return expression if isinstance(expression, str) else ""


def _rich_text(value: object) -> str:
    """Join plain text from a Notion rich-text array."""
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if isinstance(item, dict):
            plain_text = item.get("plain_text")
            if isinstance(plain_text, str):
                parts.append(plain_text)
    return "".join(parts)


def _pagination(payload: JSONObject) -> tuple[bool, str]:
    """Read and validate Notion pagination metadata."""
    has_more = payload.get("has_more")
    if not isinstance(has_more, bool):
        raise NotionApiError("invalid_response")
    cursor = payload.get("next_cursor")
    if cursor is None and not has_more:
        return False, ""
    if not isinstance(cursor, str) or (has_more and not cursor):
        raise NotionApiError("invalid_response")
    return has_more, cursor


def _non_empty_string(value: object, name: str) -> str:
    """Validate and normalize a required string argument."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Notion {name} must be a non-empty string.")
    return value.strip()


def _optional_cursor(value: object) -> str | None:
    """Validate an optional opaque Notion cursor without modifying it."""
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("Notion cursor must be a non-empty string.")
    return value


def _bounded_int(value: object, name: str, *, maximum: int) -> int:
    """Validate an integer argument with inclusive bounds."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Notion {name} must be an integer.")
    if value < 1 or value > maximum:
        raise ValueError(f"Notion {name} must be between 1 and {maximum}.")
    return value


def _required_object_list(payload: JSONObject, key: str) -> list[JSONObject]:
    """Read a required list of JSON objects from a Notion response."""
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise NotionApiError("invalid_response")
    return cast(list[JSONObject], value)


def _required_string(payload: JSONObject, key: str) -> str:
    """Read a required non-empty string from a Notion response."""
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise NotionApiError("invalid_response")
    return value


def _string_field(payload: JSONObject, key: str) -> str:
    """Return a string field or an empty string."""
    value = payload.get(key)
    return value if isinstance(value, str) else ""
