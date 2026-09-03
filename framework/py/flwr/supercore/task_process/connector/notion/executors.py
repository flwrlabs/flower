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
"""Notion action executors."""

import requests

from flwr.supercore.typing import JSONObject

from ..definition import ConnectorExecutionContext, ConnectorExecutor
from ..http import ConnectorApiError, request_json_object
from ..json_utils import ConnectorInputError, optional_string, require_int_range

_NOTION_API_BASE_URL = "https://api.notion.com/v1"
NOTION_API_VERSION = "2026-03-11"


class NotionApiError(ConnectorApiError):
    """Secret-safe Notion API failure."""

    provider = "Notion"


def search(arguments: JSONObject, context: ConnectorExecutionContext) -> JSONObject:
    """Search pages and data sources shared with the Notion connection."""
    query = arguments.get("query")
    if not isinstance(query, str):
        raise ConnectorInputError("Notion query must be a string.")
    body: JSONObject = {"query": query}
    for name in ("filter", "sort"):
        value = arguments.get(name)
        if value is not None:
            if not isinstance(value, dict):
                raise ConnectorInputError(f"Notion {name} must be an object.")
            body[name] = value
    if "pageSize" in arguments:
        body["page_size"] = require_int_range(
            arguments["pageSize"], "Notion", "pageSize", maximum=100
        )
    if cursor := optional_string(arguments.get("startCursor"), "Notion", "startCursor"):
        body["start_cursor"] = cursor
    return _call_notion_api("POST", "/search", context.credentials, body=body)


def get_page(arguments: JSONObject, context: ConnectorExecutionContext) -> JSONObject:
    """Get a Notion page together with its first-level child blocks."""
    page_id = optional_string(arguments.get("pageId"), "Notion", "pageId")
    if page_id is None:
        raise ConnectorInputError("Notion pageId must be a non-empty string.")
    page = _call_notion_api("GET", f"/pages/{page_id}", context.credentials)
    block_children = _call_notion_api(
        "GET", f"/blocks/{page_id}/children", context.credentials
    )
    return {"page": page, "block_children": block_children}


EXECUTORS: dict[str, ConnectorExecutor] = {
    "search": search,
    "get_page": get_page,
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
