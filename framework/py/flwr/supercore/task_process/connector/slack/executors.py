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
from ..http import ConnectorApiError
from ..json_utils import (
    optional_string,
    require_bool,
    require_int_range,
    require_string,
)
from .actions import SLACK_CONVERSATION_TYPES


class SlackApiError(ConnectorApiError):
    """Secret-safe Slack Web API failure."""

    provider = "Slack"


def search_messages(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Search messages visible to the connected Slack user."""
    return _call_slack_api(
        "search.messages",
        context,
        {
            "query": require_string(arguments.get("query"), "Slack", "query"),
            "count": _limit(arguments, default=5, maximum=15),
        },
    )


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
    return _call_slack_api(
        "conversations.list",
        context,
        {
            "limit": _limit(arguments, default=10, maximum=50),
            "cursor": optional_string(arguments.get("cursor"), "Slack", "cursor"),
            "types": ",".join(dict.fromkeys(selected_types)),
            "exclude_archived": str(
                require_bool(
                    arguments.get("exclude_archived", True),
                    "Slack",
                    "exclude_archived",
                )
            ).lower(),
        },
    )


def get_conversation_history(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Read one page of a Slack conversation's message history."""
    return _call_slack_api(
        "conversations.history",
        context,
        _conversation_params(arguments),
    )


def get_thread_replies(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Read one page of replies from a Slack thread."""
    params = _conversation_params(arguments)
    params["ts"] = require_string(arguments.get("thread_ts"), "Slack", "thread_ts")
    return _call_slack_api("conversations.replies", context, params)


EXECUTORS: dict[str, ConnectorExecutor] = {
    "search_messages": search_messages,
    "list_conversations": list_conversations,
    "get_conversation_history": get_conversation_history,
    "get_thread_replies": get_thread_replies,
}


def _call_slack_api(
    method: str, context: ConnectorExecutionContext, params: dict[str, str | None]
) -> JSONObject:
    """Call one Slack Web API method and validate its response envelope."""
    if context.http is None:
        raise RuntimeError("Slack HTTP client is not configured.")
    payload = context.http.request(
        "GET",
        f"/{method}",
        error=SlackApiError,
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


def _conversation_params(arguments: JSONObject) -> dict[str, str | None]:
    """Build validated parameters for a Slack conversation read."""
    return {
        "channel": require_string(
            arguments.get("conversation_id"), "Slack", "conversation_id"
        ),
        "limit": _limit(arguments, default=10, maximum=15),
        "cursor": optional_string(arguments.get("cursor"), "Slack", "cursor"),
    }


def _limit(arguments: JSONObject, *, default: int, maximum: int) -> str:
    """Return one validated Slack page limit."""
    return str(
        require_int_range(
            arguments.get("limit", default), "Slack", "limit", maximum=maximum
        )
    )
