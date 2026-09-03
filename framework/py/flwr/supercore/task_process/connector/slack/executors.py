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

from flwr.supercore.typing import JSONObject

from ..definition import ConnectorExecutionContext, ConnectorExecutor
from ..http import ConnectorApiError, request_json_object
from ..json_utils import (
    ConnectorInputError,
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
        raise ConnectorInputError("Slack page and cursor cannot be used together.")
    params: dict[str, str | None] = {
        "query": require_string(arguments.get("query"), "Slack", "query")
    }
    for name in ("count", "page"):
        if name in arguments:
            params[name] = str(
                require_int_range(
                    arguments[name], "Slack", name, minimum=1, maximum=100
                )
            )
    for argument_name, api_name in (
        ("cursor", "cursor"),
        ("sort", "sort"),
        ("sortDir", "sort_dir"),
        ("teamId", "team_id"),
    ):
        if argument_name in arguments:
            params[api_name] = _optional_raw_string(arguments, argument_name)
    if params.get("sort") not in {None, "score", "timestamp"}:
        raise ConnectorInputError("Slack sort must be 'score' or 'timestamp'.")
    if params.get("sort_dir") not in {None, "asc", "desc"}:
        raise ConnectorInputError("Slack sortDir must be 'asc' or 'desc'.")
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
    response_metadata = payload.get("response_metadata")
    return {
        "query": _string(payload.get("query")) or params["query"] or "",
        "matches": [
            _normalize_search_match(match)
            for match in (matches if isinstance(matches, list) else [])
            if isinstance(match, dict)
        ],
        "total": _integer(message_data.get("total")) or 0,
        "pagination": _object(message_data.get("pagination")),
        "paging": _object(message_data.get("paging")),
        "nextCursor": _next_cursor(response_metadata),
    }


def list_conversations(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """List conversations visible to the connected Slack user."""
    types = arguments.get("types")
    if types is not None and (
        not isinstance(types, list) or not all(isinstance(item, str) for item in types)
    ):
        raise ConnectorInputError("Slack conversation types are invalid.")
    selected_types = (
        list(SLACK_CONVERSATION_TYPES) if types is None else cast(list[str], types)
    )
    if not selected_types or any(
        item not in SLACK_CONVERSATION_TYPES for item in selected_types
    ):
        raise ConnectorInputError("Slack conversation types are invalid.")
    payload = _call_slack_api(
        "conversations.list",
        context.credentials,
        {
            "limit": _limit(arguments, default=200, maximum=200),
            "cursor": (
                _optional_raw_string(arguments, "cursor")
                if "cursor" in arguments
                else None
            ),
            "types": ",".join(dict.fromkeys(selected_types)),
            "exclude_archived": (
                str(
                    require_bool(
                        arguments["excludeArchived"], "Slack", "excludeArchived"
                    )
                ).lower()
                if "excludeArchived" in arguments
                else None
            ),
        },
    )
    channels = payload.get("channels")
    return {
        "conversations": [
            _normalize_conversation(channel)
            for channel in (channels if isinstance(channels, list) else [])
            if isinstance(channel, dict)
        ],
        "nextCursor": _next_cursor(payload.get("response_metadata")),
    }


def get_channel_messages(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Get recent messages from a Slack conversation."""
    params: dict[str, str | None] = {
        "channel": require_string(arguments.get("channelId"), "Slack", "channelId")
    }
    if "limit" in arguments:
        params["limit"] = str(
            require_int_range(
                arguments["limit"], "Slack", "limit", minimum=1, maximum=100
            )
        )
    payload = _call_slack_api(
        "conversations.history",
        context.credentials,
        params,
    )
    return _messages_output(payload)


def get_thread(arguments: JSONObject, context: ConnectorExecutionContext) -> JSONObject:
    """Get messages in a Slack thread."""
    params: dict[str, str | None] = {
        "channel": require_string(arguments.get("channelId"), "Slack", "channelId"),
        "ts": require_string(arguments.get("threadTs"), "Slack", "threadTs"),
    }
    return _messages_output(
        _call_slack_api("conversations.replies", context.credentials, params)
    )


EXECUTORS: dict[str, ConnectorExecutor] = {
    "get_channel_messages": get_channel_messages,
    "search_messages": search_messages,
    "get_thread": get_thread,
    "list_conversations": list_conversations,
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
        http_error_code=lambda response: (
            "rate_limited" if response.status_code == 429 else "http_error"
        ),
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


def _limit(arguments: JSONObject, *, default: int, maximum: int) -> str:
    """Return one validated Slack page limit."""
    return str(
        require_int_range(
            arguments.get("limit", default), "Slack", "limit", maximum=maximum
        )
    )


def _optional_raw_string(arguments: JSONObject, name: str) -> str:
    """Return an optional string exactly as supplied."""
    value = arguments[name]
    if not isinstance(value, str):
        raise ConnectorInputError(f"Slack {name} must be a string.")
    return value


def _messages_output(payload: JSONObject) -> JSONObject:
    """Normalize a Slack message-list response like Open Connector."""
    messages = payload.get("messages")
    return {
        "messages": [
            {
                "ts": _string(message.get("ts")),
                "userId": _string(message.get("user")),
                "text": _string(message.get("text")),
            }
            for message in (messages if isinstance(messages, list) else [])
            if isinstance(message, dict)
        ],
        "hasMore": payload.get("has_more") is True,
    }


def _normalize_search_match(match: JSONObject) -> JSONObject:
    """Normalize one Slack search result like Open Connector."""
    channel = _object(match.get("channel"))
    normalized: JSONObject = {
        "channelName": (
            channel.get("name") if isinstance(channel.get("name"), str) else None
        ),
        "text": _string(match.get("text")),
    }
    for output_name, value in (
        ("matchId", match.get("iid")),
        ("channelId", channel.get("id")),
        ("ts", match.get("ts")),
        ("userId", match.get("user")),
        ("username", match.get("username")),
        ("permalink", match.get("permalink")),
        ("teamId", match.get("team")),
        ("type", match.get("type")),
    ):
        if isinstance(value, str) and value.strip():
            normalized[output_name] = value.strip()
    return normalized


def _normalize_conversation(conversation: JSONObject) -> JSONObject:
    """Normalize one Slack conversation like Open Connector."""
    topic = _object(conversation.get("topic"))
    purpose = _object(conversation.get("purpose"))
    normalized: JSONObject = {
        "channelId": str(conversation.get("id") or ""),
        "name": (
            conversation.get("name")
            if isinstance(conversation.get("name"), str)
            else None
        ),
        "type": _conversation_type(conversation),
        "isArchived": _optional_bool(conversation.get("is_archived")),
        "isPrivate": _optional_bool(conversation.get("is_private")),
        "isMember": _optional_bool(conversation.get("is_member")),
        "topic": topic.get("value") if isinstance(topic.get("value"), str) else None,
        "purpose": (
            purpose.get("value") if isinstance(purpose.get("value"), str) else None
        ),
    }
    if (member_count := _integer(conversation.get("num_members"))) is not None:
        normalized["memberCount"] = member_count
    for output_name, input_name in (("userId", "user"), ("locale", "locale")):
        value = conversation.get(input_name)
        if isinstance(value, str) and value.strip():
            normalized[output_name] = value.strip()
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


def _next_cursor(value: object) -> str | None:
    """Read a non-empty Slack continuation cursor."""
    metadata = _object(value)
    cursor = metadata.get("next_cursor")
    return cursor if isinstance(cursor, str) and cursor else None


def _object(value: object) -> JSONObject:
    """Return a JSON object or an empty object."""
    return value if isinstance(value, dict) else {}


def _string(value: object) -> str:
    """Return a string or an empty string."""
    return value if isinstance(value, str) else ""


def _integer(value: object) -> int | None:
    """Return an integer or None."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_bool(value: object) -> bool | None:
    """Return a boolean or None."""
    return value if isinstance(value, bool) else None
