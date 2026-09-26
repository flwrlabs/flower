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

from collections.abc import Mapping
from typing import cast
from urllib.parse import quote, unquote

import requests

from flwr.supercore.typing import JSONObject

from ..definition import ConnectorExecutionContext, ConnectorExecutor
from ..http import ConnectorApiError, request_json_object
from ..json_utils import optional_string, require_int_range, require_string

_NOTION_API_BASE_URL = "https://api.notion.com/v1"
NOTION_API_VERSION = "2026-03-11"


class NotionApiError(ConnectorApiError):
    """Secret-safe Notion API failure."""

    provider = "Notion"


def search(arguments: JSONObject, context: ConnectorExecutionContext) -> JSONObject:
    """Search pages and data sources shared with the Notion connection."""
    body: JSONObject = {}
    if "query" in arguments:
        body["query"] = require_string(arguments["query"], "Notion", "query")
    if "filter" in arguments:
        body["filter"] = _search_filter(arguments["filter"])
    if "sort" in arguments:
        body["sort"] = arguments["sort"]
    if "page_size" in arguments:
        body["page_size"] = require_int_range(
            arguments["page_size"], "Notion", "page_size", maximum=100
        )
    if cursor := optional_string(
        arguments.get("start_cursor"), "Notion", "start_cursor"
    ):
        body["start_cursor"] = cursor
    return _call_notion_api("POST", "/search", context.credentials, body=body)


def _search_filter(value: object) -> JSONObject:
    """Validate a Notion search filter without changing it."""
    if not isinstance(value, dict):
        raise ValueError("Notion filter must be an object.")
    keys = set(value)
    trash_only = keys == {"in_trash"} and isinstance(value["in_trash"], bool)
    object_filter = (
        {"property", "value"} <= keys <= {"property", "value", "in_trash"}
        and value["property"] == "object"
        and isinstance(value["value"], str)
        and value["value"] in {"page", "data_source"}
        and ("in_trash" not in value or isinstance(value["in_trash"], bool))
    )
    if not trash_only and not object_filter:
        raise ValueError("Notion filter is invalid.")
    return cast(JSONObject, value)


def get_page(arguments: JSONObject, context: ConnectorExecutionContext) -> JSONObject:
    """Retrieve one Notion page and its property values."""
    page_id = require_string(arguments.get("page_id"), "Notion", "page_id")
    params: dict[str, str | list[str]] = {}
    if "filter_properties" in arguments:
        params["filter_properties"] = _property_ids(arguments["filter_properties"])
    return _call_notion_api(
        "GET",
        f"/pages/{quote(page_id, safe='')}",
        context.credentials,
        params=params,
    )


def _property_ids(value: object) -> list[str]:
    """Validate property IDs used to filter a Notion page response."""
    if not isinstance(value, list) or len(value) > 100:
        raise ValueError(
            "Notion filter_properties must be an array of at most 100 IDs."
        )
    return [
        unquote(require_string(item, "Notion", "filter_properties item"))
        for item in value
    ]


def get_page_property(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Retrieve one property value from a Notion page."""
    page_id = require_string(arguments.get("page_id"), "Notion", "page_id")
    property_id = require_string(arguments.get("property_id"), "Notion", "property_id")
    params: dict[str, str] = {}
    if "page_size" in arguments:
        params["page_size"] = str(
            require_int_range(
                arguments["page_size"], "Notion", "page_size", maximum=100
            )
        )
    if cursor := optional_string(
        arguments.get("start_cursor"), "Notion", "start_cursor"
    ):
        params["start_cursor"] = cursor
    return _call_notion_api(
        "GET",
        "/pages/"
        f"{quote(page_id, safe='')}/properties/"
        f"{quote(unquote(property_id), safe='')}",
        context.credentials,
        params=params,
    )


def list_users(arguments: JSONObject, context: ConnectorExecutionContext) -> JSONObject:
    """List workspace users."""
    params: dict[str, str] = {}
    if "page_size" in arguments:
        params["page_size"] = str(
            require_int_range(
                arguments["page_size"], "Notion", "page_size", maximum=100
            )
        )
    if cursor := optional_string(
        arguments.get("start_cursor"), "Notion", "start_cursor"
    ):
        params["start_cursor"] = cursor
    return _call_notion_api("GET", "/users", context.credentials, params=params)


def get_user(arguments: JSONObject, context: ConnectorExecutionContext) -> JSONObject:
    """Retrieve one workspace user by ID."""
    user_id = require_string(arguments.get("user_id"), "Notion", "user_id")
    return _call_notion_api(
        "GET", f"/users/{quote(user_id, safe='')}", context.credentials
    )


def get_self(_arguments: JSONObject, context: ConnectorExecutionContext) -> JSONObject:
    """Retrieve the user associated with the access token."""
    return _call_notion_api("GET", "/users/me", context.credentials)


EXECUTORS: dict[str, ConnectorExecutor] = {
    "search": search,
    "get_page": get_page,
    "get_page_property": get_page_property,
    "list_users": list_users,
    "get_user": get_user,
    "get_self": get_self,
}


def _call_notion_api(
    method: str,
    path: str,
    credentials: JSONObject,
    *,
    body: JSONObject | None = None,
    params: Mapping[str, str | list[str]] | None = None,
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
        http_error_details=_response_error_details,
    )


def _response_error_details(response: requests.Response) -> tuple[str, str | None]:
    """Return Notion's documented error code and message."""
    try:
        payload = response.json()
    except ValueError:
        return "http_error", None
    if not isinstance(payload, dict):
        return "http_error", None
    code = payload.get("code")
    message = payload.get("message")
    return (
        code if isinstance(code, str) and code else "http_error",
        message if isinstance(message, str) and message else None,
    )
