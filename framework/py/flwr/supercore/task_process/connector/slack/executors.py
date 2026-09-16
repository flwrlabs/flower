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
"""Slack action executors."""

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
)
from .actions import SLACK_CONVERSATION_TYPES

_SLACK_API_BASE_URL = "https://slack.com/api"


class SlackApiError(ConnectorApiError):
    """Secret-safe Slack Web API failure."""

    provider = "Slack"


def search_messages(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Search messages visible to the connected Slack user."""
    if arguments.get("page") is not None and arguments.get("cursor") is not None:
        raise ValueError("Slack page and cursor cannot be used together.")
    params: dict[str, str | None] = {
        "query": require_string(arguments.get("query"), "Slack", "query"),
        "cursor": optional_string(arguments.get("cursor"), "Slack", "cursor"),
        "sort": optional_string(arguments.get("sort"), "Slack", "sort"),
        "sort_dir": optional_string(arguments.get("sort_dir"), "Slack", "sort_dir"),
        "team_id": optional_string(arguments.get("team_id"), "Slack", "team_id"),
    }
    for name in ("count", "page"):
        if name in arguments:
            params[name] = str(
                require_int_range(
                    arguments[name], "Slack", name, minimum=1, maximum=100
                )
            )
    if "highlight" in arguments:
        params["highlight"] = str(
            require_bool(arguments["highlight"], "Slack", "highlight")
        ).lower()
    payload = _call_slack_api(
        "search.messages",
        context.credentials,
        params,
    )
    messages = payload.get("messages")
    message_data = messages if isinstance(messages, dict) else {}
    matches = message_data.get("matches")
    metadata = payload.get("response_metadata")
    return {
        "query": (
            payload.get("query")
            if isinstance(payload.get("query"), str)
            else params["query"]
        ),
        "matches": (
            [
                _normalize_search_match(match)
                for match in matches
                if isinstance(match, dict)
            ]
            if isinstance(matches, list)
            else []
        ),
        "total": (
            message_data.get("total")
            if isinstance(message_data.get("total"), int)
            else 0
        ),
        "pagination": (
            message_data.get("pagination")
            if isinstance(message_data.get("pagination"), dict)
            else {}
        ),
        "paging": (
            message_data.get("paging")
            if isinstance(message_data.get("paging"), dict)
            else {}
        ),
        "next_cursor": _next_cursor(metadata),
    }


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
        "limit": str(
            require_int_range(
                arguments.get("limit", 200), "Slack", "limit", maximum=200
            )
        ),
        "cursor": optional_string(arguments.get("cursor"), "Slack", "cursor"),
        "types": ",".join(dict.fromkeys(selected_types)),
    }
    if "exclude_archived" in arguments:
        params["exclude_archived"] = str(
            require_bool(arguments["exclude_archived"], "Slack", "exclude_archived")
        ).lower()
    payload = _call_slack_api(
        "conversations.list",
        context.credentials,
        params,
    )
    channels = payload.get("channels")
    return {
        "conversations": (
            [
                _normalize_conversation(channel)
                for channel in channels
                if isinstance(channel, dict)
            ]
            if isinstance(channels, list)
            else []
        ),
        "next_cursor": _next_cursor(payload.get("response_metadata")),
    }


def get_channel_messages(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Get recent messages from a Slack conversation."""
    params: dict[str, str | None] = {
        "channel": require_string(arguments.get("channel_id"), "Slack", "channel_id")
    }
    if "limit" in arguments:
        params["limit"] = str(
            require_int_range(arguments["limit"], "Slack", "limit", maximum=100)
        )
    payload = _call_slack_api(
        "conversations.history",
        context.credentials,
        params,
    )
    return _normalize_messages(payload)


def get_thread(arguments: JSONObject, context: ConnectorExecutionContext) -> JSONObject:
    """Get messages in a Slack thread."""
    payload = _call_slack_api(
        "conversations.replies",
        context.credentials,
        {
            "channel": require_string(
                arguments.get("channel_id"), "Slack", "channel_id"
            ),
            "ts": require_string(arguments.get("thread_ts"), "Slack", "thread_ts"),
        },
    )
    return _normalize_messages(payload)


EXECUTORS: dict[str, ConnectorExecutor] = {
    "search_messages": search_messages,
    "list_conversations": list_conversations,
    "get_channel_messages": get_channel_messages,
    "get_thread": get_thread,
}


def _call_slack_api(
    method: str, credentials: JSONObject, params: dict[str, str | None]
) -> JSONObject:
    """Call one Slack Web API method and validate its response envelope."""
    token = credentials.get("access_token")
    if not isinstance(token, str) or not token:
        raise SlackApiError("invalid_credentials")
    payload = request_json_object(
        "GET",
        f"{_SLACK_API_BASE_URL}/{method}",
        error=SlackApiError,
        headers={"Authorization": f"Bearer {token}"},
        params={key: value for key, value in params.items() if value is not None},
        http_error_details=_response_error_details,
    )
    if payload.get("ok") is not True:
        error = payload.get("error")
        code = (
            error
            if isinstance(error, str)
            and error.replace("_", "").isalnum()
            and error.islower()
            else "api_error"
        )
        raise SlackApiError(code)
    return payload


def _response_error_details(response: requests.Response) -> tuple[str, str | None]:
    """Return Slack's documented error code and message."""
    fallback_code = "rate_limited" if response.status_code == 429 else "http_error"
    try:
        payload = response.json()
    except ValueError:
        return fallback_code, None
    if not isinstance(payload, dict):
        return fallback_code, None
    code = payload.get("error")
    message = payload.get("message")
    return (
        code if isinstance(code, str) and code else fallback_code,
        message if isinstance(message, str) and message else None,
    )


def _normalize_messages(payload: JSONObject) -> JSONObject:
    """Normalize a Slack message-list response."""
    messages = payload.get("messages")
    return {
        "messages": (
            [
                {
                    "ts": _string(message.get("ts")),
                    "user_id": _string(message.get("user")),
                    "text": _string(message.get("text")),
                }
                for message in messages
                if isinstance(message, dict)
            ]
            if isinstance(messages, list)
            else []
        ),
        "has_more": payload.get("has_more") is True,
    }


def _normalize_search_match(match: JSONObject) -> JSONObject:
    """Normalize one Slack message search match using snake_case fields."""
    channel = match.get("channel")
    channel_data = channel if isinstance(channel, dict) else {}
    normalized: JSONObject = {
        "channel_name": (
            channel_data.get("name")
            if isinstance(channel_data.get("name"), str)
            else None
        ),
        "text": _string(match.get("text")),
    }
    for output_name, value in (
        ("match_id", match.get("iid")),
        ("channel_id", channel_data.get("id")),
        ("ts", match.get("ts")),
        ("user_id", match.get("user")),
        ("username", match.get("username")),
        ("permalink", match.get("permalink")),
        ("team_id", match.get("team")),
        ("type", match.get("type")),
    ):
        if isinstance(value, str):
            normalized[output_name] = value
    return normalized


def _normalize_conversation(conversation: JSONObject) -> JSONObject:
    """Normalize one Slack conversation using snake_case fields."""
    topic = conversation.get("topic")
    purpose = conversation.get("purpose")
    normalized: JSONObject = {
        "channel_id": _string(conversation.get("id")),
        "name": (
            conversation.get("name")
            if isinstance(conversation.get("name"), str)
            else None
        ),
        "type": _conversation_type(conversation),
        "is_archived": _optional_bool(conversation.get("is_archived")),
        "is_private": _optional_bool(conversation.get("is_private")),
        "is_member": _optional_bool(conversation.get("is_member")),
        "topic": (
            topic.get("value")
            if isinstance(topic, dict) and isinstance(topic.get("value"), str)
            else None
        ),
        "purpose": (
            purpose.get("value")
            if isinstance(purpose, dict) and isinstance(purpose.get("value"), str)
            else None
        ),
    }
    for name, value in (
        ("member_count", conversation.get("num_members")),
        ("user_id", conversation.get("user")),
        ("locale", conversation.get("locale")),
    ):
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            normalized[name] = value
    return normalized


def _conversation_type(conversation: JSONObject) -> str:
    """Return Open Connector's normalized Slack conversation type."""
    if conversation.get("is_im") is True:
        return "im"
    if conversation.get("is_mpim") is True:
        return "mpim"
    if conversation.get("is_private") is True or conversation.get("is_group") is True:
        return "private_channel"
    if conversation.get("is_channel") is True:
        return "public_channel"
    return "unknown"


def _next_cursor(metadata: object) -> str | None:
    """Read Slack's next cursor from response metadata."""
    if not isinstance(metadata, dict):
        return None
    cursor = metadata.get("next_cursor")
    return cursor if isinstance(cursor, str) and cursor else None


def _optional_bool(value: object) -> bool | None:
    """Return a boolean value or None."""
    return value if isinstance(value, bool) else None


def _string(value: object) -> str:
    """Return a string value or an empty string."""
    return value if isinstance(value, str) else ""
