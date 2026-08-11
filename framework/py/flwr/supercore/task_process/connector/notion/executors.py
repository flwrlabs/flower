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
from ..http import ConnectorApiError
from ..json_utils import optional_string, require_int_range, require_string

NOTION_API_VERSION = "2026-03-11"


class NotionApiError(ConnectorApiError):
    """Secret-safe Notion API failure."""

    provider = "Notion"


def search(arguments: JSONObject, context: ConnectorExecutionContext) -> JSONObject:
    """Search pages and data sources shared with the Notion connection."""
    body: JSONObject = {
        "query": require_string(arguments.get("query"), "Notion", "query"),
        "page_size": require_int_range(
            arguments.get("limit", 10), "Notion", "limit", maximum=100
        ),
    }
    if cursor := optional_string(arguments.get("cursor"), "Notion", "cursor"):
        body["start_cursor"] = cursor
    return _call_notion_api("POST", "/search", context, body=body)


def get_page_content(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Read one page of a Notion page's block content."""
    params = {
        "page_size": str(
            require_int_range(
                arguments.get("max_blocks", 100),
                "Notion",
                "max_blocks",
                maximum=100,
            )
        )
    }
    if cursor := optional_string(arguments.get("cursor"), "Notion", "cursor"):
        params["start_cursor"] = cursor
    page_id = require_string(arguments.get("page_id"), "Notion", "page_id")
    return _call_notion_api(
        "GET", f"/blocks/{page_id}/children", context, params=params
    )


EXECUTORS: dict[str, ConnectorExecutor] = {
    "search": search,
    "get_page_content": get_page_content,
}


def _call_notion_api(
    method: str,
    path: str,
    context: ConnectorExecutionContext,
    *,
    body: JSONObject | None = None,
    params: dict[str, str] | None = None,
) -> JSONObject:
    """Call one Notion API endpoint and return its JSON response."""
    if context.http is None:
        raise RuntimeError("Notion HTTP client is not configured.")
    return context.http.request(
        method,
        path,
        error=NotionApiError,
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
