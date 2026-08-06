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
from .json_utils import optional_string, require_bool, require_int_range, require_string
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
_CONVERSATION_TYPES = ("public_channel", "private_channel", "mpim", "im")
_CURSOR = string_property("Cursor returned by the previous Slack response.")
_MESSAGE_LIMIT = integer_property(
    "Maximum number of messages to return.", minimum=1, maximum=15
)

SLACK_TOOLS = (
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
                "Maximum number of conversations to return.", minimum=1, maximum=50
            ),
            "cursor": _CURSOR,
            "types": {
                "type": "array",
                "items": {"type": "string", "enum": list(_CONVERSATION_TYPES)},
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
            "limit": _MESSAGE_LIMIT,
            "cursor": _CURSOR,
        },
        required=("conversation_id",),
    ),
    function_tool(
        SLACK_GET_THREAD_REPLIES_TOOL,
        "Read a Slack thread's parent message and replies.",
        properties={
            "conversation_id": string_property("Slack conversation ID."),
            "thread_ts": string_property("Timestamp of the thread's parent message."),
            "limit": _MESSAGE_LIMIT,
            "cursor": _CURSOR,
        },
        required=("conversation_id", "thread_ts"),
    ),
)


class SlackApiError(ConnectorApiError):
    """Secret-safe Slack Web API failure."""

    provider = "Slack"


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
    return _call_slack_api(
        "search.messages",
        credentials,
        {
            "query": require_string(query, "Slack", "query"),
            "count": str(require_int_range(limit, "Slack", "limit", maximum=15)),
        },
    )


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
    selected_types = list(_CONVERSATION_TYPES) if types is None else types
    if not selected_types or any(
        item not in _CONVERSATION_TYPES for item in selected_types
    ):
        raise ValueError("Slack conversation types are invalid.")
    return _call_slack_api(
        "conversations.list",
        credentials,
        {
            "limit": str(require_int_range(limit, "Slack", "limit", maximum=50)),
            "cursor": optional_string(cursor, "Slack", "cursor"),
            "types": ",".join(dict.fromkeys(selected_types)),
            "exclude_archived": str(
                require_bool(exclude_archived, "Slack", "exclude_archived")
            ).lower(),
        },
    )


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
    return _call_slack_api(
        "conversations.history",
        credentials,
        _conversation_params(conversation_id, limit, cursor),
    )


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
    params = _conversation_params(conversation_id, limit, cursor)
    params["ts"] = require_string(thread_ts, "Slack", "thread_ts")
    return _call_slack_api("conversations.replies", credentials, params)


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


def _conversation_params(
    conversation_id: str, limit: int, cursor: str | None
) -> dict[str, str | None]:
    """Build validated parameters for a Slack conversation read."""
    return {
        "channel": require_string(conversation_id, "Slack", "conversation_id"),
        "limit": str(require_int_range(limit, "Slack", "limit", maximum=15)),
        "cursor": optional_string(cursor, "Slack", "cursor"),
    }
