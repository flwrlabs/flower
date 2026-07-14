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
"""Read-only Slack connector tools."""

import re
from collections.abc import Callable
from typing import cast

import requests

from flwr.supercore.task_process.usage import TaskUsageRecorder
from flwr.supercore.typing import JSONObject, JSONValue

SLACK_CONNECTOR_REF = "slack"
SLACK_SEARCH_MESSAGES_TOOL = "slack_search_messages"
SLACK_LIST_CONVERSATIONS_TOOL = "slack_list_conversations"
SLACK_GET_CONVERSATION_HISTORY_TOOL = "slack_get_conversation_history"
SLACK_GET_THREAD_REPLIES_TOOL = "slack_get_thread_replies"

SLACK_TOOL_NAMES = (
    SLACK_SEARCH_MESSAGES_TOOL,
    SLACK_LIST_CONVERSATIONS_TOOL,
    SLACK_GET_CONVERSATION_HISTORY_TOOL,
    SLACK_GET_THREAD_REPLIES_TOOL,
)

_SLACK_API_BASE_URL = "https://slack.com/api"
_REQUEST_TIMEOUT = 30.0
_SAFE_ERROR_CODE = re.compile(r"^[a-z0-9_]+$")


class SlackApiError(RuntimeError):
    """Secret-safe Slack Web API failure."""

    def __init__(self, code: str, status_code: int | None = None) -> None:
        self.code = code
        self.status_code = status_code
        detail = code if status_code is None else f"{code} ({status_code})"
        super().__init__(f"Slack API request failed: {detail}.")


def make_slack_tools() -> list[JSONObject]:
    """Return model-facing schemas for Slack's read-only v1 operations."""
    cursor: JSONObject = {
        "cursor": {
            "type": "string",
            "description": "Cursor returned by the previous Slack response.",
        }
    }
    return [
        {
            "type": "function",
            "name": SLACK_SEARCH_MESSAGES_TOOL,
            "description": "Search messages visible to the connected Slack user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Slack message search query.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 15,
                        "description": "Maximum number of matches to return.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": SLACK_LIST_CONVERSATIONS_TOOL,
            "description": "List Slack channels and direct-message conversations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "description": "Maximum number of conversations to return.",
                    },
                    **cursor,
                    "types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "public_channel",
                                "private_channel",
                                "mpim",
                                "im",
                            ],
                        },
                        "description": "Conversation types to include.",
                    },
                    "exclude_archived": {
                        "type": "boolean",
                        "description": "Whether to exclude archived conversations.",
                    },
                },
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": SLACK_GET_CONVERSATION_HISTORY_TOOL,
            "description": "Read recent messages from one Slack conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "conversation_id": {
                        "type": "string",
                        "description": "Slack conversation ID.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 15,
                        "description": "Maximum number of messages to return.",
                    },
                    **cursor,
                },
                "required": ["conversation_id"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": SLACK_GET_THREAD_REPLIES_TOOL,
            "description": "Read a Slack thread's parent message and replies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "conversation_id": {
                        "type": "string",
                        "description": "Slack conversation ID containing the thread.",
                    },
                    "thread_ts": {
                        "type": "string",
                        "description": "Timestamp of the thread's parent message.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 15,
                        "description": "Maximum number of messages to return.",
                    },
                    **cursor,
                },
                "required": ["conversation_id", "thread_ts"],
                "additionalProperties": False,
            },
        },
    ]


def search_messages(
    query: str,
    limit: int = 5,
    *,
    credentials: JSONObject,
    config: JSONObject,
    usage_recorder: TaskUsageRecorder,
) -> JSONObject:
    """Search messages visible to the connected Slack user."""
    del config, usage_recorder
    payload = _call_slack_api(
        "search.messages",
        credentials,
        {
            "query": _non_empty_string(query, "query"),
            "count": str(_bounded_int(limit, "limit", maximum=15)),
        },
    )
    messages = _required_object(payload, "messages")
    matches = _required_object_list(messages, "matches")
    return {"results": [_normalize_search_match(match) for match in matches[:limit]]}


# pylint: disable-next=too-many-arguments
def list_conversations(
    limit: int = 10,
    cursor: str | None = None,
    types: list[str] | None = None,
    exclude_archived: bool = True,
    *,
    credentials: JSONObject,
    config: JSONObject,
    usage_recorder: TaskUsageRecorder,
) -> JSONObject:
    """List conversations visible to the connected Slack user."""
    del config, usage_recorder
    limit = _bounded_int(limit, "limit", maximum=50)
    selected_types = _conversation_types(types)
    payload = _call_slack_api(
        "conversations.list",
        credentials,
        {
            "limit": str(limit),
            "cursor": _optional_non_empty_string(cursor, "cursor"),
            "types": ",".join(selected_types),
            "exclude_archived": str(
                _boolean(exclude_archived, "exclude_archived")
            ).lower(),
        },
    )
    channels = _required_object_list(payload, "channels")
    return {
        "conversations": [
            _normalize_conversation(channel) for channel in channels[:limit]
        ],
        "next_cursor": _next_cursor(payload),
    }


# pylint: disable-next=too-many-arguments
def get_conversation_history(
    conversation_id: str,
    limit: int = 10,
    cursor: str | None = None,
    *,
    credentials: JSONObject,
    config: JSONObject,
    usage_recorder: TaskUsageRecorder,
) -> JSONObject:
    """Read one page of a Slack conversation's message history."""
    del config, usage_recorder
    payload = _call_slack_api(
        "conversations.history",
        credentials,
        _conversation_params(
            conversation_id=conversation_id,
            limit=limit,
            cursor=cursor,
        ),
    )
    return _normalize_message_page(payload, limit=limit)


# pylint: disable-next=too-many-arguments
def get_thread_replies(
    conversation_id: str,
    thread_ts: str,
    limit: int = 10,
    cursor: str | None = None,
    *,
    credentials: JSONObject,
    config: JSONObject,
    usage_recorder: TaskUsageRecorder,
) -> JSONObject:
    """Read one page of replies from a Slack thread."""
    del config, usage_recorder
    thread_ts = _non_empty_string(thread_ts, "thread_ts")
    params = _conversation_params(
        conversation_id=conversation_id,
        limit=limit,
        cursor=cursor,
    )
    params["ts"] = thread_ts
    payload = _call_slack_api("conversations.replies", credentials, params)
    normalized = _normalize_message_page(payload, limit=limit)
    normalized["thread_ts"] = thread_ts
    return normalized


SLACK_TOOL_HANDLERS: dict[str, Callable[..., JSONValue]] = {
    SLACK_SEARCH_MESSAGES_TOOL: search_messages,
    SLACK_LIST_CONVERSATIONS_TOOL: list_conversations,
    SLACK_GET_CONVERSATION_HISTORY_TOOL: get_conversation_history,
    SLACK_GET_THREAD_REPLIES_TOOL: get_thread_replies,
}


def _call_slack_api(
    method: str, credentials: JSONObject, params: dict[str, str | None]
) -> JSONObject:
    """Call one Slack Web API method and validate its response envelope."""
    access_token = credentials.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise SlackApiError("invalid_credentials")
    request_params = {key: value for key, value in params.items() if value is not None}
    try:
        response = requests.get(
            f"{_SLACK_API_BASE_URL}/{method}",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            params=request_params,
            timeout=_REQUEST_TIMEOUT,
        )
    except requests.RequestException:
        raise SlackApiError("request_failed") from None
    if response.status_code == 429:
        raise SlackApiError("rate_limited", status_code=429)
    if response.status_code >= 400:
        raise SlackApiError("http_error", status_code=response.status_code)
    try:
        payload = response.json()
    except ValueError:
        raise SlackApiError("invalid_response") from None
    if not isinstance(payload, dict):
        raise SlackApiError("invalid_response")
    if payload.get("ok") is not True:
        error = payload.get("error")
        code = (
            error
            if isinstance(error, str) and _SAFE_ERROR_CODE.fullmatch(error)
            else "api_error"
        )
        raise SlackApiError(code)
    return cast(JSONObject, payload)


def _conversation_params(
    *, conversation_id: str, limit: int, cursor: str | None
) -> dict[str, str | None]:
    """Build validated shared parameters for Slack conversation reads."""
    return {
        "channel": _non_empty_string(conversation_id, "conversation_id"),
        "limit": str(_bounded_int(limit, "limit", maximum=15)),
        "cursor": _optional_non_empty_string(cursor, "cursor"),
    }


def _conversation_types(types: object) -> list[str]:
    """Validate Slack conversation type filters."""
    if types is None:
        return ["public_channel", "private_channel", "mpim", "im"]
    if not isinstance(types, list) or not types:
        raise ValueError("Slack conversation types must be a non-empty list.")
    allowed = {"public_channel", "private_channel", "mpim", "im"}
    selected: list[str] = []
    for item in types:
        if not isinstance(item, str) or item not in allowed:
            raise ValueError("Slack conversation types are invalid.")
        if item not in selected:
            selected.append(item)
    return selected


def _non_empty_string(value: object, name: str) -> str:
    """Validate and normalize a required string argument."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Slack {name} must be a non-empty string.")
    return value.strip()


def _optional_non_empty_string(value: object, name: str) -> str | None:
    """Validate and normalize an optional string argument."""
    if value is None:
        return None
    return _non_empty_string(value, name)


def _bounded_int(value: object, name: str, *, maximum: int) -> int:
    """Validate an integer argument with inclusive bounds."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Slack {name} must be an integer.")
    if value < 1 or value > maximum:
        raise ValueError(f"Slack {name} must be between 1 and {maximum}.")
    return value


def _boolean(value: object, name: str) -> bool:
    """Validate a boolean argument."""
    if not isinstance(value, bool):
        raise ValueError(f"Slack {name} must be a boolean.")
    return value


def _required_object(payload: JSONObject, key: str) -> JSONObject:
    """Read a required JSON object from a Slack response."""
    value = payload.get(key)
    if not isinstance(value, dict):
        raise SlackApiError("invalid_response")
    return value


def _required_object_list(payload: JSONObject, key: str) -> list[JSONObject]:
    """Read a required list of JSON objects from a Slack response."""
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise SlackApiError("invalid_response")
    return cast(list[JSONObject], value)


def _next_cursor(payload: JSONObject) -> str:
    """Read Slack's next cursor from a response envelope."""
    metadata = payload.get("response_metadata")
    if metadata is None:
        return ""
    if not isinstance(metadata, dict):
        raise SlackApiError("invalid_response")
    cursor = metadata.get("next_cursor")
    if cursor in (None, ""):
        return ""
    if not isinstance(cursor, str):
        raise SlackApiError("invalid_response")
    return cursor


def _normalize_search_match(message: JSONObject) -> JSONObject:
    """Return the stable subset of a Slack search match."""
    channel = message.get("channel")
    channel_id = ""
    channel_name = ""
    channel_type = ""
    if isinstance(channel, dict):
        channel_id = _string_field(channel, "id")
        channel_name = _string_field(channel, "name") or _string_field(channel, "user")
        if channel.get("is_channel") is True:
            channel_type = "public_channel"
        elif channel.get("is_group") is True:
            channel_type = "private_channel"
        elif channel.get("is_im") is True:
            channel_type = "im"
        elif channel.get("is_mpim") is True:
            channel_type = "mpim"
    return {
        "conversation_id": channel_id,
        "conversation_name": channel_name,
        "conversation_type": channel_type,
        "user": _string_field(message, "username") or _string_field(message, "user"),
        "ts": _string_field(message, "ts"),
        "text": _string_field(message, "text"),
        "permalink": _string_field(message, "permalink"),
    }


def _normalize_conversation(channel: JSONObject) -> JSONObject:
    """Return the stable subset of a Slack conversation."""
    topic = channel.get("topic")
    purpose = channel.get("purpose")
    return {
        "id": _string_field(channel, "id"),
        "name": _string_field(channel, "name") or _string_field(channel, "user"),
        "is_channel": channel.get("is_channel") is True,
        "is_private": channel.get("is_private") is True,
        "is_im": channel.get("is_im") is True,
        "is_mpim": channel.get("is_mpim") is True,
        "is_archived": channel.get("is_archived") is True,
        "num_members": _int_field(channel, "num_members"),
        "topic": _string_field(topic, "value") if isinstance(topic, dict) else "",
        "purpose": (
            _string_field(purpose, "value") if isinstance(purpose, dict) else ""
        ),
    }


def _normalize_message_page(payload: JSONObject, *, limit: int) -> JSONObject:
    """Normalize a Slack message page and cursor metadata."""
    messages = _required_object_list(payload, "messages")
    return {
        "messages": [_normalize_message(message) for message in messages[:limit]],
        "has_more": payload.get("has_more") is True,
        "next_cursor": _next_cursor(payload),
    }


def _normalize_message(message: JSONObject) -> JSONObject:
    """Return the stable subset of a Slack message."""
    return {
        "type": _string_field(message, "type"),
        "subtype": _string_field(message, "subtype"),
        "user": _string_field(message, "user"),
        "text": _string_field(message, "text"),
        "ts": _string_field(message, "ts"),
        "thread_ts": _string_field(message, "thread_ts"),
        "parent_user_id": _string_field(message, "parent_user_id"),
        "reply_count": _int_field(message, "reply_count"),
    }


def _string_field(payload: JSONObject, key: str) -> str:
    """Return a string field or an empty string."""
    value = payload.get(key)
    return value if isinstance(value, str) else ""


def _int_field(payload: JSONObject, key: str) -> int | None:
    """Return an integer field or None."""
    value = payload.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None
