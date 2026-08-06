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
"""Tests for read-only Slack connector tools."""

from collections.abc import Callable
from unittest.mock import Mock, patch

import pytest
import requests

from flwr.supercore.task_process.agent.session import RuntimeAgentConnectors
from flwr.supercore.typing import JSONObject

from . import registry
from .slack import (
    SLACK_CONNECTOR_REF,
    SLACK_SEARCH_MESSAGES_TOOL,
    SLACK_TOOL_NAMES,
    SlackApiError,
    get_conversation_history,
    get_thread_replies,
    list_conversations,
    make_slack_tools,
    search_messages,
)

_CREDENTIALS = {"access_token": "xoxp-secret"}


def test_slack_tools_are_registered_separately_from_builtins() -> None:
    """Slack should expose multiple tools without becoming a built-in connector."""
    tools = registry.get_connector_tools(SLACK_CONNECTOR_REF)

    assert [tool["name"] for tool in tools] == list(SLACK_TOOL_NAMES)
    assert registry.has_builtin_connector(SLACK_CONNECTOR_REF) is False
    assert registry.get_connector_ref(SLACK_SEARCH_MESSAGES_TOOL) == "slack"
    assert registry.get_connector_ref("web_search") == "web_search"
    assert registry.get_connector_ref("unknown_tool") == "unknown_tool"


def test_agent_connector_selection_expands_slack_tools() -> None:
    """Selecting Slack should advertise all Slack tools to the model."""
    connectors = RuntimeAgentConnectors(Mock())

    tools = connectors.tools(["web_search", "slack"])

    assert [tool["name"] for tool in tools] == ["web_search", *SLACK_TOOL_NAMES]


def test_slack_tool_schemas_are_closed_read_only_functions() -> None:
    """Every Slack tool should have a closed schema and no write operation."""
    tools = make_slack_tools()

    assert {tool["name"] for tool in tools} == set(SLACK_TOOL_NAMES)
    for tool in tools:
        assert tool["type"] == "function"
        assert tool["parameters"]["additionalProperties"] is False
        assert "write" not in str(tool["name"])
        assert "send" not in str(tool["name"])


def test_search_messages_returns_normalized_matches() -> None:
    """Message search should return the stable subset of Slack match fields."""
    response = Mock(status_code=200)
    response.json.return_value = {
        "ok": True,
        "messages": {
            "matches": [
                {
                    "ts": "1.0",
                    "text": "release",
                    "username": "ada",
                    "permalink": "https://flower.slack.com/archives/C123/p1",
                    "channel": {
                        "id": "C123",
                        "name": "team-agent",
                        "is_channel": True,
                    },
                }
            ]
        },
    }
    with patch(
        "flwr.supercore.task_process.connector.http.requests.request",
        return_value=response,
    ) as get:
        result = search_messages(
            " release notes ",
            limit=5,
            credentials=_CREDENTIALS,
            config={},
            usage_recorder=Mock(),
        )

    get.assert_called_once_with(
        "GET",
        "https://slack.com/api/search.messages",
        headers={
            "Authorization": "Bearer xoxp-secret",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        params={"query": "release notes", "count": "5"},
        json=None,
        timeout=30.0,
    )
    assert result == {
        "results": [
            {
                "conversation_id": "C123",
                "conversation_name": "team-agent",
                "conversation_type": "public_channel",
                "user": "ada",
                "ts": "1.0",
                "text": "release",
                "permalink": "https://flower.slack.com/archives/C123/p1",
            }
        ]
    }


def test_list_conversations_normalizes_results_and_cursor() -> None:
    """Conversation listing should return stable fields and cursor pagination."""
    response = Mock(status_code=200)
    response.json.return_value = {
        "ok": True,
        "channels": [
            {
                "id": "C123",
                "name": "team-agent",
                "is_channel": True,
                "is_private": False,
                "is_im": False,
                "is_mpim": False,
                "is_archived": False,
                "num_members": 42,
                "topic": {"value": "Agent development"},
                "purpose": {"value": "Build agents"},
            }
        ],
        "response_metadata": {"next_cursor": "next-page"},
    }
    with patch(
        "flwr.supercore.task_process.connector.http.requests.request",
        return_value=response,
    ) as get:
        result = list_conversations(
            limit=25,
            cursor="current-page",
            types=["public_channel", "private_channel", "public_channel"],
            exclude_archived=True,
            credentials=_CREDENTIALS,
            config={},
            usage_recorder=Mock(),
        )

    assert get.call_args.kwargs["params"] == {
        "limit": "25",
        "cursor": "current-page",
        "types": "public_channel,private_channel",
        "exclude_archived": "true",
    }
    assert result == {
        "conversations": [
            {
                "id": "C123",
                "name": "team-agent",
                "is_channel": True,
                "is_private": False,
                "is_im": False,
                "is_mpim": False,
                "is_archived": False,
                "num_members": 42,
                "topic": "Agent development",
                "purpose": "Build agents",
            }
        ],
        "next_cursor": "next-page",
    }


@pytest.mark.parametrize(
    ("function", "method", "extra_arguments", "extra_params", "extra_result"),
    [
        (
            get_conversation_history,
            "conversations.history",
            {},
            {},
            {},
        ),
        (
            get_thread_replies,
            "conversations.replies",
            {"thread_ts": "100.200"},
            {"ts": "100.200"},
            {"thread_ts": "100.200"},
        ),
    ],
)
def test_conversation_reads_normalize_messages_and_pagination(
    function: Callable[..., JSONObject],
    method: str,
    extra_arguments: dict[str, str],
    extra_params: dict[str, str],
    extra_result: dict[str, str],
) -> None:
    """History and thread tools should return stable message pages."""
    response = Mock(status_code=200)
    response.json.return_value = {
        "ok": True,
        "messages": [
            {
                "type": "message",
                "user": "U123",
                "text": "parent",
                "ts": "100.200",
                "thread_ts": "100.200",
                "reply_count": 2,
            }
        ],
        "has_more": True,
        "response_metadata": {"next_cursor": "next-page"},
    }
    with patch(
        "flwr.supercore.task_process.connector.http.requests.request",
        return_value=response,
    ) as get:
        result = function(
            conversation_id="C123",
            limit=15,
            cursor="current-page",
            credentials=_CREDENTIALS,
            config={},
            usage_recorder=Mock(),
            **extra_arguments,
        )

    assert get.call_args.args == ("GET", f"https://slack.com/api/{method}")
    assert get.call_args.kwargs["params"] == {
        "channel": "C123",
        "limit": "15",
        "cursor": "current-page",
        **extra_params,
    }
    assert result == {
        "messages": [
            {
                "type": "message",
                "subtype": "",
                "user": "U123",
                "text": "parent",
                "ts": "100.200",
                "thread_ts": "100.200",
                "parent_user_id": "",
                "reply_count": 2,
            }
        ],
        "has_more": True,
        "next_cursor": "next-page",
        **extra_result,
    }


@pytest.mark.parametrize("tool_name", SLACK_TOOL_NAMES)
def test_registry_maps_every_slack_tool_to_one_connection(tool_name: str) -> None:
    """All Slack tool calls should resolve the account's single Slack credential."""
    assert registry.requires_connector_credentials(tool_name)
    assert registry.get_connector_ref(tool_name) == SLACK_CONNECTOR_REF


def test_slack_api_errors_are_stable_and_secret_safe() -> None:
    """Slack response details and bearer tokens must not leak through errors."""
    response = Mock(status_code=200)
    response.json.return_value = {
        "ok": False,
        "error": "invalid_auth",
        "detail": "xoxp-secret",
    }
    with (
        patch(
            "flwr.supercore.task_process.connector.http.requests.request",
            return_value=response,
        ),
        pytest.raises(SlackApiError) as error,
    ):
        search_messages(
            "release",
            credentials=_CREDENTIALS,
            config={},
            usage_recorder=Mock(),
        )

    assert error.value.code == "invalid_auth"
    assert "xoxp-secret" not in str(error.value)


@pytest.mark.parametrize(
    ("response", "side_effect", "expected_code"),
    [
        (None, requests.RequestException("xoxp-secret"), "request_failed"),
        (Mock(status_code=429), None, "rate_limited"),
        (Mock(status_code=500), None, "http_error"),
        (
            Mock(status_code=200, **{"json.side_effect": ValueError()}),
            None,
            "invalid_response",
        ),
    ],
)
def test_slack_transport_failures_are_mapped_without_details(
    response: Mock | None,
    side_effect: Exception | None,
    expected_code: str,
) -> None:
    """Transport, HTTP, and decoding failures should use stable error codes."""
    with (
        patch(
            "flwr.supercore.task_process.connector.http.requests.request",
            return_value=response,
            side_effect=side_effect,
        ),
        pytest.raises(SlackApiError) as error,
    ):
        search_messages(
            "release",
            credentials=_CREDENTIALS,
            config={},
            usage_recorder=Mock(),
        )

    assert error.value.code == expected_code
    assert "xoxp-secret" not in str(error.value)


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (search_messages, {"query": ""}),
        (search_messages, {"query": "valid", "limit": 16}),
        (list_conversations, {"types": ["invalid"]}),
        (list_conversations, {"types": []}),
        (list_conversations, {"exclude_archived": "false"}),
        (get_conversation_history, {"conversation_id": "", "limit": 10}),
    ],
)
def test_slack_tool_inputs_are_validated(
    function: Callable[..., JSONObject], arguments: dict[str, object]
) -> None:
    """Invalid tool inputs should fail before making an HTTP request."""
    with (
        patch(
            "flwr.supercore.task_process.connector.http.requests.request"
        ) as get,
        pytest.raises(ValueError),
    ):
        function(
            **arguments,
            credentials=_CREDENTIALS,
            config={},
            usage_recorder=Mock(),
        )

    get.assert_not_called()
