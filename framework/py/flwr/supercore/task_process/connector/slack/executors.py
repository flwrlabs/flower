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

from datetime import UTC, datetime
from typing import cast

import requests

from flwr.supercore.typing import JSONObject

from ..definition import ConnectorExecutionContext, ConnectorExecutor
from ..http import ConnectorApiError, request_json_object
from ..json_utils import (
    optional_string,
    require_bool,
    require_int_range,
    require_string,
    string_field,
)
from .actions import (
    SLACK_CONVERSATION_TYPES,
    SLACK_LIST_CONVERSATIONS_MAX_LIMIT,
    SLACK_MESSAGE_MAX_LIMIT,
)

_SLACK_API_BASE_URL = "https://slack.com/api"
_FILE_MAX_BYTES = 10 * 1024 * 1024
_SEARCH_CHANNEL_TYPES = ("public_channel", "private_channel", "mpim", "im")
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
    arguments: JSONObject, name: str, default: tuple[str, ...], allowed: tuple[str, ...]
) -> list[str]:
    """Parse one comma-separated MCP-style option for a Web API request."""
    raw = optional_string(arguments.get(name), "Slack", name)
    values = list(default) if raw is None else [part.strip() for part in raw.split(",")]
    if not values or any(value not in allowed for value in values):
        raise ValueError(f"Slack {name} contains an unsupported value.")
    return list(dict.fromkeys(values))


def _search_query(arguments: JSONObject) -> str:
    """Build the lexical query accepted by Slack's search APIs."""
    query = optional_string(arguments.get("query"), "Slack", "query")
    keywords = arguments.get("keywords")
    if keywords is not None and (
        not isinstance(keywords, list)
        or any(not isinstance(term, str) or not term.strip() for term in keywords)
    ):
        raise ValueError("Slack keywords must be an array of nonempty strings.")
    natural_language_query = arguments.get("natural_language_query")
    if natural_language_query is not None and not isinstance(
        natural_language_query, str
    ):
        raise ValueError("Slack natural_language_query must be a string.")
    filters = optional_string(arguments.get("filters"), "Slack", "filters")
    terms = ([query] if query else []) + list(cast(list[str], keywords or []))
    if filters:
        terms.append(filters)
    if not terms:
        raise ValueError("Slack search requires query, keywords, or filters.")
    return " ".join(terms)


def _search(
    arguments: JSONObject,
    context: ConnectorExecutionContext,
    *,
    content_types: tuple[str, ...],
    channel_types: tuple[str, ...] = _SEARCH_CHANNEL_TYPES,
) -> JSONObject:
    """Search using Slack's Real-time Search Web API."""
    payload = _search_payload(arguments, content_types, channel_types)
    try:
        result = _call_slack_api(
            "assistant.search.context", context.credentials, body=payload
        )
    except SlackApiError as error:
        if error.code not in (
            "feature_not_enabled",
            "assistant_search_context_disabled",
            "missing_scope",
        ):
            raise
        result = _search_web_api_fallback(
            arguments, context.credentials, content_types, channel_types
        )
    if "only_my_channels" in arguments and require_bool(
        arguments["only_my_channels"], "Slack", "only_my_channels"
    ):
        _filter_joined_channels(result, context.credentials)
    if "max_context_length" in arguments:
        _truncate_search_context(
            result,
            require_int_range(
                arguments["max_context_length"],
                "Slack",
                "max_context_length",
                maximum=100_000,
            ),
        )
    if arguments.get("response_format") == "concise":
        _concise_search_result(result)
    return result


def _search_payload(
    arguments: JSONObject,
    content_types: tuple[str, ...],
    channel_types: tuple[str, ...],
) -> JSONObject:
    """Translate the shared search options to Real-time Search parameters."""
    query = _search_query(arguments)
    keywords = cast(list[str], arguments.get("keywords") or [])
    natural_language_query = arguments.get("natural_language_query")
    if (
        isinstance(natural_language_query, str)
        and natural_language_query.strip()
        and len(keywords) <= 5
    ):
        filters = optional_string(arguments.get("filters"), "Slack", "filters")
        query = " ".join(part for part in (natural_language_query, filters) if part)
    payload: JSONObject = {
        "query": query,
        "content_types": (
            list(content_types)
            if content_types in (("channels",), ("users",))
            else _csv(arguments, "content_types", content_types, _SEARCH_CONTENT_TYPES)
        ),
    }
    if keywords and len(keywords) <= 5:
        payload["term_clauses"] = keywords
    if content_types == ("channels",):
        payload["channel_types"] = _csv(
            arguments,
            "channel_types",
            ("public_channel",),
            ("public_channel", "private_channel"),
        )
    elif content_types != ("users",):
        payload["channel_types"] = (
            ["public_channel"]
            if channel_types == ("public_channel",)
            else _csv(arguments, "channel_types", channel_types, _SEARCH_CHANNEL_TYPES)
        )
    for name in ("context_channel_id", "cursor", "sort", "sort_dir"):
        value = optional_string(arguments.get(name), "Slack", name)
        if value is not None:
            payload[name] = value
    if "limit" in arguments:
        payload["limit"] = require_int_range(
            arguments["limit"], "Slack", "limit", maximum=20
        )
    for name in ("after", "before"):
        value = optional_string(arguments.get(name), "Slack", name)
        if value is not None:
            try:
                payload[name] = int(value)
            except ValueError:
                raise ValueError(f"Slack {name} must be a Unix timestamp.") from None
    if "include_bots" in arguments:
        payload["include_bots"] = require_bool(
            arguments["include_bots"], "Slack", "include_bots"
        )
    payload["include_context_messages"] = require_bool(
        arguments.get("include_context", True), "Slack", "include_context"
    )
    if "include_archived" in arguments:
        payload["include_archived_channels"] = require_bool(
            arguments["include_archived"], "Slack", "include_archived"
        )
    return payload


def search_public(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Search public-channel messages and files."""
    return _search(
        arguments,
        context,
        content_types=_SEARCH_CONTENT_TYPES,
        channel_types=("public_channel",),
    )


def search_public_and_private(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Search visible messages and files in all conversation types."""
    return _search(arguments, context, content_types=_SEARCH_CONTENT_TYPES)


def _search_web_api_fallback(
    arguments: JSONObject,
    credentials: JSONObject,
    content_types: tuple[str, ...],
    channel_types: tuple[str, ...],
) -> JSONObject:
    """Search messages and files through the standard Web API fallback."""
    return _search_messages_files_fallback(arguments, credentials, channel_types)


def _search_messages_files_fallback(
    arguments: JSONObject,
    credentials: JSONObject,
    channel_types: tuple[str, ...],
) -> JSONObject:
    """Search messages and files through the standard search.all method."""
    query = _search_query(arguments)
    limit = require_int_range(arguments.get("limit", 20), "Slack", "limit", maximum=20)
    selected_content = _csv(
        arguments, "content_types", _SEARCH_CONTENT_TYPES, _SEARCH_CONTENT_TYPES
    )
    allowed_channels = (
        ["public_channel"]
        if channel_types == ("public_channel",)
        else _csv(arguments, "channel_types", channel_types, _SEARCH_CHANNEL_TYPES)
    )
    for name in ("after", "before"):
        value = optional_string(arguments.get(name), "Slack", name)
        if value is not None:
            try:
                date = datetime.fromtimestamp(int(value), tz=UTC).date().isoformat()
            except (ValueError, OverflowError):
                raise ValueError(f"Slack {name} must be a Unix timestamp.") from None
            query += f" {name}:{date}"
    params = {"query": query, "count": str(limit)}
    for name in ("cursor", "sort", "sort_dir"):
        value = optional_string(arguments.get(name), "Slack", name)
        if value is not None:
            params[name] = value
    page = _call_slack_api("search.all", credentials, params=params)
    results: JSONObject = {}
    if "messages" in selected_content:
        results["messages"] = _legacy_messages(
            page, allowed_channels, arguments.get("include_bots") is True
        )
    if "files" in selected_content:
        results["files"] = _legacy_files(page, allowed_channels)
    return {
        "ok": True,
        "results": results,
        "response_metadata": page.get("response_metadata", {}),
    }


def _legacy_messages(
    page: JSONObject, allowed_channels: list[str], include_bots: bool
) -> list[JSONObject]:
    """Select visible message results from search.all."""
    message_page = page.get("messages")
    matches = message_page.get("matches") if isinstance(message_page, dict) else []
    if not isinstance(matches, list):
        raise SlackApiError("invalid_response")
    messages: list[JSONObject] = []
    for item in matches:
        if not isinstance(item, dict):
            continue
        channel = item.get("channel")
        if not isinstance(channel, dict):
            continue
        if _message_channel_type(channel) not in allowed_channels:
            continue
        if not include_bots and isinstance(item.get("bot_id"), str):
            continue
        messages.append(
            {
                "channel_id": channel.get("id"),
                "message_ts": item.get("ts"),
                "author_user_id": item.get("user"),
                "content": item.get("text"),
                "permalink": item.get("permalink"),
            }
        )
    return messages


def _legacy_files(page: JSONObject, allowed_channels: list[str]) -> list[JSONObject]:
    """Select files shared in the permitted conversation types."""
    file_page = page.get("files")
    matches = file_page.get("matches") if isinstance(file_page, dict) else []
    if not isinstance(matches, list):
        raise SlackApiError("invalid_response")
    files: list[JSONObject] = []
    for item in matches:
        if not isinstance(item, dict):
            continue
        channel_id = _file_channel_id(item, allowed_channels)
        if channel_id is None:
            continue
        files.append(
            {
                "file_id": item.get("id"),
                "title": item.get("title"),
                "content": item.get("preview"),
                "permalink": item.get("permalink"),
                "channel_id": channel_id,
            }
        )
    return files


def _message_channel_type(channel: JSONObject) -> str:
    """Classify a search result's channel for visibility filtering."""
    channel_id = channel.get("id")
    if channel.get("is_im") is True:
        kind = "im"
    elif channel.get("is_mpim") is True:
        kind = "mpim"
    elif channel.get("is_private") is True:
        kind = "private_channel"
    elif isinstance(channel_id, str) and channel_id.startswith("D"):
        kind = "im"
    elif isinstance(channel_id, str) and channel_id.startswith("G"):
        kind = "private_channel"
    elif isinstance(channel_id, str) and channel_id.startswith("C"):
        kind = "public_channel"
    else:
        kind = ""
    return kind


def _file_channel_id(file: JSONObject, allowed: list[str]) -> str | None:
    """Find a file share in one of the permitted channel types."""
    fields = (
        ("public_channel", "channels"),
        ("private_channel", "groups"),
        ("mpim", "mpims"),
        ("im", "ims"),
    )
    for channel_type, field in fields:
        channels = file.get(field)
        if channel_type in allowed and isinstance(channels, list):
            for channel_id in channels:
                if isinstance(channel_id, str):
                    return channel_id
    return None


def _filter_joined_channels(result: JSONObject, credentials: JSONObject) -> None:
    """Keep search results from conversations the user has joined."""
    joined: set[str] = set()
    cursor = ""
    while True:
        params = {"types": ",".join(_SEARCH_CHANNEL_TYPES), "limit": "200"}
        if cursor:
            params["cursor"] = cursor
        page = _call_slack_api("users.conversations", credentials, params=params)
        channels = page.get("channels")
        if not isinstance(channels, list):
            raise SlackApiError("invalid_response")
        for channel in channels:
            if isinstance(channel, dict) and isinstance(channel.get("id"), str):
                joined.add(channel["id"])
        metadata = page.get("response_metadata")
        cursor = (
            string_field(metadata, "next_cursor") if isinstance(metadata, dict) else ""
        )
        if not cursor:
            break
    results = result.get("results")
    if not isinstance(results, dict):
        return
    for content_type in ("messages", "files", "channels"):
        items = results.get(content_type)
        if isinstance(items, list):
            results[content_type] = [
                item
                for item in items
                if isinstance(item, dict)
                and item.get("channel_id", item.get("id")) in joined
            ]


def _truncate_search_context(result: JSONObject, maximum: int) -> None:
    """Truncate surrounding message text to the requested length."""
    results = result.get("results")
    messages = results.get("messages") if isinstance(results, dict) else None
    if not isinstance(messages, list):
        return
    for message in messages:
        context = message.get("context_messages") if isinstance(message, dict) else None
        if isinstance(context, dict):
            _truncate_context_entries(context, maximum)


def _truncate_context_entries(context: JSONObject, maximum: int) -> None:
    """Shorten text in the messages surrounding one search result."""
    for direction in ("before", "after"):
        surrounding = context.get(direction)
        if not isinstance(surrounding, list):
            continue
        for item in surrounding:
            if not isinstance(item, dict):
                continue
            for field in ("text", "content"):
                value = item.get(field)
                if isinstance(value, str):
                    item[field] = value[:maximum]


def _concise_search_result(result: JSONObject) -> None:
    """Reduce search entries to identifiers, content, and links."""
    results = result.get("results")
    if not isinstance(results, dict):
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
        "channels": ("channel_id", "id", "name", "topic", "purpose"),
        "users": ("user_id", "id", "name", "real_name", "email"),
    }
    for kind, names in fields.items():
        items = results.get(kind)
        if isinstance(items, list):
            results[kind] = [
                {key: item[key] for key in names if key in item}
                for item in items
                if isinstance(item, dict)
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
    fallback_code = "rate_limited" if response.status_code == 429 else "http_error"
    try:
        payload = response.json()
    except ValueError:
        return fallback_code, None
    if not isinstance(payload, dict):
        return fallback_code, None
    return _payload_error_details(cast(JSONObject, payload), fallback_code)


def _payload_error_details(
    payload: JSONObject, fallback_code: str
) -> tuple[str, str | None]:
    """Return error details from either Slack response envelope shape."""
    error = payload.get("error")
    code = error if isinstance(error, str) and error else fallback_code
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
