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

from collections.abc import Callable

from flwr.supercore.task_process.usage import TaskUsageRecorder
from flwr.supercore.typing import JSONObject, JSONValue

from .http import ConnectorApiError, request_json_object
from .json_utils import (
    integer_field,
    object_field,
    object_list_field,
    optional_string,
    require_bool,
    require_int_range,
    require_string,
    string_field,
)
from .tool_schema import function_tool, integer_property, string_property

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
class SlackApiError(ConnectorApiError):
    """Secret-safe Slack Web API failure."""

    provider = "Slack"


def make_slack_tools() -> list[JSONObject]:
    """Return model-facing schemas for Slack's read-only v1 operations."""
    cursor = string_property("Cursor returned by the previous Slack response.")
    return [
        function_tool(
            SLACK_SEARCH_MESSAGES_TOOL,
            "Search messages visible to the connected Slack user.",
            properties={
                "query": string_property("Slack message search query."),
                "limit": integer_property(
                    "Maximum number of matches to return.", minimum=1, maximum=15
                ),
            },
            required=("query",),
        ),
        function_tool(
            SLACK_LIST_CONVERSATIONS_TOOL,
            "List Slack channels and direct-message conversations.",
            properties={
                "limit": integer_property(
                    "Maximum number of conversations to return.",
                    minimum=1,
                    maximum=50,
                ),
                "cursor": cursor,
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
        ),
        function_tool(
            SLACK_GET_CONVERSATION_HISTORY_TOOL,
            "Read recent messages from one Slack conversation.",
            properties={
                "conversation_id": string_property("Slack conversation ID."),
                "limit": integer_property(
                    "Maximum number of messages to return.", minimum=1, maximum=15
                ),
                "cursor": cursor,
            },
            required=("conversation_id",),
        ),
        function_tool(
            SLACK_GET_THREAD_REPLIES_TOOL,
            "Read a Slack thread's parent message and replies.",
            properties={
                "conversation_id": string_property(
                    "Slack conversation ID containing the thread."
                ),
                "thread_ts": string_property(
                    "Timestamp of the thread's parent message."
                ),
                "limit": integer_property(
                    "Maximum number of messages to return.", minimum=1, maximum=15
                ),
                "cursor": cursor,
            },
            required=("conversation_id", "thread_ts"),
        ),
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
            "query": require_string(query, "Slack", "query"),
            "count": str(require_int_range(limit, "Slack", "limit", maximum=15)),
        },
    )
    messages = object_field(payload, "messages", error=SlackApiError)
    matches = object_list_field(messages, "matches", error=SlackApiError)
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
    limit = require_int_range(limit, "Slack", "limit", maximum=50)
    selected_types = _conversation_types(types)
    payload = _call_slack_api(
        "conversations.list",
        credentials,
        {
            "limit": str(limit),
            "cursor": optional_string(cursor, "Slack", "cursor"),
            "types": ",".join(selected_types),
            "exclude_archived": str(
                require_bool(exclude_archived, "Slack", "exclude_archived")
            ).lower(),
        },
    )
    channels = object_list_field(payload, "channels", error=SlackApiError)
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
    thread_ts = require_string(thread_ts, "Slack", "thread_ts")
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
    payload = request_json_object(
        "GET",
        f"{_SLACK_API_BASE_URL}/{method}",
        error=SlackApiError,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        params={key: value for key, value in params.items() if value is not None},
        http_error_code=lambda response: (
            "rate_limited" if response.status_code == 429 else "http_error"
        ),
    )
    if payload.get("ok") is not True:
        error = payload.get("error")
        code = error if isinstance(error, str) and error.isidentifier() else "api_error"
        raise SlackApiError(code)
    return payload


def _conversation_params(
    *, conversation_id: str, limit: int, cursor: str | None
) -> dict[str, str | None]:
    """Build validated shared parameters for Slack conversation reads."""
    return {
        "channel": require_string(conversation_id, "Slack", "conversation_id"),
        "limit": str(require_int_range(limit, "Slack", "limit", maximum=15)),
        "cursor": optional_string(cursor, "Slack", "cursor"),
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
        channel_id = string_field(channel, "id")
        channel_name = string_field(channel, "name") or string_field(channel, "user")
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
        "user": string_field(message, "username") or string_field(message, "user"),
        "ts": string_field(message, "ts"),
        "text": string_field(message, "text"),
        "permalink": string_field(message, "permalink"),
    }


def _normalize_conversation(channel: JSONObject) -> JSONObject:
    """Return the stable subset of a Slack conversation."""
    topic = channel.get("topic")
    purpose = channel.get("purpose")
    return {
        "id": string_field(channel, "id"),
        "name": string_field(channel, "name") or string_field(channel, "user"),
        "is_channel": channel.get("is_channel") is True,
        "is_private": channel.get("is_private") is True,
        "is_im": channel.get("is_im") is True,
        "is_mpim": channel.get("is_mpim") is True,
        "is_archived": channel.get("is_archived") is True,
        "num_members": integer_field(channel, "num_members"),
        "topic": string_field(topic, "value") if isinstance(topic, dict) else "",
        "purpose": (
            string_field(purpose, "value") if isinstance(purpose, dict) else ""
        ),
    }


def _normalize_message_page(payload: JSONObject, *, limit: int) -> JSONObject:
    """Normalize a Slack message page and cursor metadata."""
    messages = object_list_field(payload, "messages", error=SlackApiError)
    return {
        "messages": [_normalize_message(message) for message in messages[:limit]],
        "has_more": payload.get("has_more") is True,
        "next_cursor": _next_cursor(payload),
    }


def _normalize_message(message: JSONObject) -> JSONObject:
    """Return the stable subset of a Slack message."""
    return {
        "type": string_field(message, "type"),
        "subtype": string_field(message, "subtype"),
        "user": string_field(message, "user"),
        "text": string_field(message, "text"),
        "ts": string_field(message, "ts"),
        "thread_ts": string_field(message, "thread_ts"),
        "parent_user_id": string_field(message, "parent_user_id"),
        "reply_count": integer_field(message, "reply_count"),
    }
