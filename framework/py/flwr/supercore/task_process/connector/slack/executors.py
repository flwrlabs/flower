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
"""Execute Slack-shaped read tools through the Slack Web API."""

from __future__ import annotations

from typing import cast

import requests

from flwr.supercore.typing import JSONObject

from ..definition import ConnectorExecutionContext, ConnectorExecutor
from ..http import ConnectorApiError, request_json_object
from ..json_utils import (
    object_field,
    object_list_field,
    optional_string,
    require_bool,
    require_int_range,
    require_string,
    required_string_field,
    string_field,
)
from .actions import (
    SLACK_CONVERSATION_TYPES,
    SLACK_LIST_CONVERSATIONS_MAX_LIMIT,
    SLACK_MESSAGE_MAX_LIMIT,
)

_SLACK_API_BASE_URL = "https://slack.com/api"
_SEARCH_CONTENT_TYPES = ("messages", "files")


class SlackApiError(ConnectorApiError):
    """Secret-safe Slack Web API failure."""

    provider = "Slack"


def _call_slack_api(
    method: str,
    credentials: JSONObject,
    *,
    params: dict[str, str] | None = None,
    body: JSONObject | None = None,
) -> JSONObject:
    """Call one Slack Web API method and validate its response."""
    token = credentials.get("access_token")
    if not isinstance(token, str) or not token:
        raise SlackApiError("invalid_credentials")
    payload = request_json_object(
        "POST" if body is not None else "GET",
        f"{_SLACK_API_BASE_URL}/{method}",
        error=SlackApiError,
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        json=body,
        http_error_details=_response_error_details,
    )
    if payload.get("ok") is not True:
        code, message = _payload_error_details(payload, "api_error")
        raise SlackApiError(code, message=message)
    return payload


def _csv(
    arguments: JSONObject,
    name: str,
    choices: tuple[str, ...],
    *,
    default: tuple[str, ...],
) -> list[str]:
    """Parse one comma-separated MCP-style option for a Web API request."""
    raw = optional_string(arguments.get(name), "Slack", name)
    values = list(default) if raw is None else [part.strip() for part in raw.split(",")]
    if not values or any(value not in choices for value in values):
        raise ValueError(f"Slack {name} contains an unsupported value.")
    return list(dict.fromkeys(values))


def _search(
    arguments: JSONObject,
    context: ConnectorExecutionContext,
    *,
    channel_types: tuple[str, ...] = SLACK_CONVERSATION_TYPES,
) -> JSONObject:
    """Search using Slack's Real-time Search Web API."""
    payload = _search_payload(arguments, channel_types)
    only_my_channels = require_bool(
        arguments.get("only_my_channels", False), "Slack", "only_my_channels"
    )
    response_format = arguments.get("response_format", "detailed")
    if response_format not in ("detailed", "concise"):
        raise ValueError("Slack response_format must be detailed or concise.")
    maximum = None
    if "max_context_length" in arguments:
        value = arguments["max_context_length"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Slack max_context_length must be a nonnegative integer.")
        maximum = value
    result = _call_slack_api(
        "assistant.search.context", context.credentials, body=payload
    )
    if only_my_channels:
        _filter_joined_channels(
            result,
            context.credentials,
            cast(list[str], payload["channel_types"]),
        )
    if maximum is not None or response_format == "concise":
        _format_search_result(result, maximum, concise=response_format == "concise")
    return result


def _search_payload(
    arguments: JSONObject,
    channel_types: tuple[str, ...],
) -> JSONObject:
    """Translate the shared search options to Real-time Search parameters."""
    query = optional_string(arguments.get("query"), "Slack", "query")
    keywords = arguments.get("keywords", [])
    if not isinstance(keywords, list) or any(
        not isinstance(term, str) or not term.strip() for term in keywords
    ):
        raise ValueError("Slack keywords must be an array of nonempty strings.")
    terms = cast(list[str], keywords)
    if len(terms) > 5:
        raise ValueError(
            "Slack search supports at most 5 keywords; reduce the keyword list."
        )
    filters = optional_string(arguments.get("filters"), "Slack", "filters")
    natural_language_query = optional_string(
        arguments.get("natural_language_query"), "Slack", "natural_language_query"
    )
    if not query and not terms and not filters:
        raise ValueError("Slack search requires query, keywords, or filters.")
    query_parts = (
        [natural_language_query, query, filters]
        if natural_language_query
        else [query, *terms, filters]
    )
    payload: JSONObject = {
        "query": " ".join(part for part in query_parts if part),
        "content_types": _csv(
            arguments, "content_types", _SEARCH_CONTENT_TYPES, default=("messages",)
        ),
        "channel_types": (
            ["public_channel"]
            if channel_types == ("public_channel",)
            else _csv(arguments, "channel_types", channel_types, default=channel_types)
        ),
    }
    if terms:
        payload["term_clauses"] = terms
        if filters:
            payload["modifiers"] = filters
    if "limit" in arguments:
        payload["limit"] = require_int_range(
            arguments["limit"], "Slack", "limit", maximum=20
        )
    for name in ("context_channel_id", "cursor", "sort", "sort_dir"):
        if name in arguments:
            payload[name] = arguments[name]
    for name in ("after", "before"):
        value = optional_string(arguments.get(name), "Slack", name)
        if value is not None:
            payload[name] = int(value)
    payload["include_bots"] = arguments.get("include_bots", False)
    payload["include_context_messages"] = arguments.get("include_context", True)
    return payload


def search_public(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Search public-channel messages and files."""
    return _search(
        arguments,
        context,
        channel_types=("public_channel",),
    )


def search_public_and_private(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Search visible messages and files in all conversation types."""
    return _search(arguments, context)


def _joined_channel_ids(credentials: JSONObject, channel_types: list[str]) -> set[str]:
    """List the selected conversation types that the connected user has joined."""
    joined: set[str] = set()
    cursor = ""
    while True:
        params = {"types": ",".join(channel_types), "limit": "200"}
        if cursor:
            params["cursor"] = cursor
        page = _call_slack_api("users.conversations", credentials, params=params)
        for channel in object_list_field(page, "channels", error=SlackApiError):
            joined.add(required_string_field(channel, "id", error=SlackApiError))
        metadata = page.get("response_metadata")
        cursor = (
            string_field(metadata, "next_cursor") if isinstance(metadata, dict) else ""
        )
        if not cursor:
            break
    return joined


def _filter_joined_channels(
    result: JSONObject, credentials: JSONObject, channel_types: list[str]
) -> None:
    """Keep search results from conversations the user has joined."""
    joined = _joined_channel_ids(credentials, channel_types)
    results = object_field(result, "results", error=SlackApiError)
    if "messages" in results:
        results["messages"] = [
            item
            for item in object_list_field(results, "messages", error=SlackApiError)
            if required_string_field(item, "channel_id", error=SlackApiError) in joined
        ]
    if "files" in results:
        files = []
        for item in object_list_field(results, "files", error=SlackApiError):
            file_id = required_string_field(item, "file_id", error=SlackApiError)
            file = object_field(
                _call_slack_api("files.info", credentials, params={"file": file_id}),
                "file",
                error=SlackApiError,
            )
            fields = ("channels", "groups", "ims")
            if not any(field in file for field in fields):
                raise SlackApiError("invalid_response")
            shared_channels: set[str] = set()
            for field in fields:
                channels = file.get(field, [])
                if not isinstance(channels, list) or not all(
                    isinstance(channel_id, str) for channel_id in channels
                ):
                    raise SlackApiError("invalid_response")
                shared_channels.update(cast(list[str], channels))
            if joined.intersection(shared_channels):
                files.append(item)
        results["files"] = files


def _truncate_search_context(messages: list[JSONObject], maximum: int) -> None:
    """Truncate surrounding message text to the requested length."""
    for message in messages:
        context = message.get("context_messages")
        if not isinstance(context, dict):
            continue
        for direction in ("before", "after"):
            if direction not in context:
                continue
            for item in object_list_field(context, direction, error=SlackApiError):
                item["text"] = string_field(item, "text")[:maximum]


def _format_search_result(
    result: JSONObject, maximum: int | None, *, concise: bool
) -> None:
    """Apply context truncation and concise formatting to search results."""
    results = object_field(result, "results", error=SlackApiError)
    if maximum is not None and "messages" in results:
        _truncate_search_context(
            object_list_field(results, "messages", error=SlackApiError), maximum
        )
    if not concise:
        return
    fields = {
        "messages": (
            "channel_id",
            "message_ts",
            "author_user_id",
            "content",
            "permalink",
        ),
        "files": ("file_id", "title", "content", "permalink"),
    }
    for kind, names in fields.items():
        if kind in results:
            results[kind] = [
                {key: item[key] for key in names if key in item}
                for item in object_list_field(results, kind, error=SlackApiError)
            ]


def list_conversations(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """List conversations visible to the connected Slack user."""
    types = arguments.get("types")
    if types is not None and (
        not isinstance(types, list) or not all(isinstance(item, str) for item in types)
    ):
        raise ValueError("Slack conversation types are invalid.")
    selected_types = (
        list(SLACK_CONVERSATION_TYPES) if types is None else cast(list[str], types)
    )
    if not selected_types or any(
        item not in SLACK_CONVERSATION_TYPES for item in selected_types
    ):
        raise ValueError("Slack conversation types are invalid.")
    params: dict[str, str | None] = {
        "cursor": optional_string(arguments.get("cursor"), "Slack", "cursor"),
        "types": ",".join(dict.fromkeys(selected_types)),
        "team_id": optional_string(arguments.get("team_id"), "Slack", "team_id"),
    }
    if "limit" in arguments:
        params["limit"] = str(
            require_int_range(
                arguments["limit"],
                "Slack",
                "limit",
                maximum=SLACK_LIST_CONVERSATIONS_MAX_LIMIT,
            )
        )
    if "exclude_archived" in arguments:
        params["exclude_archived"] = str(
            require_bool(arguments["exclude_archived"], "Slack", "exclude_archived")
        ).lower()
    return _call_slack_api(
        "conversations.list",
        context.credentials,
        params={key: value for key, value in params.items() if value is not None},
    )


def get_conversation_history(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Get recent messages from a Slack conversation."""
    params: dict[str, str | None] = {
        "channel": require_string(arguments.get("channel_id"), "Slack", "channel_id"),
        "cursor": optional_string(arguments.get("cursor"), "Slack", "cursor"),
    }
    if "limit" in arguments:
        params["limit"] = str(
            require_int_range(
                arguments["limit"],
                "Slack",
                "limit",
                maximum=SLACK_MESSAGE_MAX_LIMIT,
            )
        )
    return _call_slack_api(
        "conversations.history",
        context.credentials,
        params={key: value for key, value in params.items() if value is not None},
    )


def get_conversation_replies(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Get messages in a Slack thread."""
    params: dict[str, str | None] = {
        "channel": require_string(arguments.get("channel_id"), "Slack", "channel_id"),
        "ts": require_string(arguments.get("thread_ts"), "Slack", "thread_ts"),
        "cursor": optional_string(arguments.get("cursor"), "Slack", "cursor"),
    }
    if "limit" in arguments:
        params["limit"] = str(
            require_int_range(
                arguments["limit"],
                "Slack",
                "limit",
                maximum=SLACK_MESSAGE_MAX_LIMIT,
            )
        )
    return _call_slack_api(
        "conversations.replies",
        context.credentials,
        params={key: value for key, value in params.items() if value is not None},
    )


EXECUTORS: dict[str, ConnectorExecutor] = {
    "search_public": search_public,
    "search_public_and_private": search_public_and_private,
    "list_conversations": list_conversations,
    "get_conversation_history": get_conversation_history,
    "get_conversation_replies": get_conversation_replies,
}


def _response_error_details(response: requests.Response) -> tuple[str, str | None]:
    """Return Slack's documented error code and message."""
    default_code = "rate_limited" if response.status_code == 429 else "http_error"
    try:
        payload = response.json()
    except ValueError:
        return default_code, None
    if not isinstance(payload, dict):
        return default_code, None
    return _payload_error_details(cast(JSONObject, payload), default_code)


def _payload_error_details(
    payload: JSONObject, default_code: str
) -> tuple[str, str | None]:
    """Return error details from either Slack response envelope shape."""
    error = payload.get("error")
    code = error if isinstance(error, str) and error else default_code
    message = payload.get("message")
    if isinstance(message, str) and message:
        return code, message
    metadata = payload.get("response_metadata")
    messages = metadata.get("messages") if isinstance(metadata, dict) else None
    if isinstance(messages, list):
        diagnostics = [item for item in messages if isinstance(item, str) and item]
        if diagnostics:
            return code, "; ".join(diagnostics)
    return code, None
