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
"""Tests for read-only Notion connector tools."""

from collections.abc import Callable
from unittest.mock import Mock, patch

import pytest
import requests

from flwr.supercore.task_process.agent.session import RuntimeAgentConnectors
from flwr.supercore.typing import JSONObject

from . import registry
from .notion import (
    NOTION_CONNECTOR_REF,
    NOTION_SEARCH_TOOL,
    NOTION_TOOL_NAMES,
    NotionApiError,
    get_page_content,
    make_notion_tools,
    search,
)

_CREDENTIALS = {"access_token": "ntn-secret"}


def test_notion_tools_are_registered_separately_from_builtins() -> None:
    """Notion should expose multiple tools without becoming a built-in."""
    tools = registry.get_connector_tools(NOTION_CONNECTOR_REF)

    assert [tool["name"] for tool in tools] == list(NOTION_TOOL_NAMES)
    assert registry.has_builtin_connector(NOTION_CONNECTOR_REF) is False
    assert registry.get_connector_ref(NOTION_SEARCH_TOOL) == "notion"


def test_agent_connector_selection_expands_notion_tools() -> None:
    """Selecting Notion should advertise all Notion tools to the model."""
    connectors = RuntimeAgentConnectors(Mock())

    tools = connectors.tools(["web_search", "notion"])

    assert [tool["name"] for tool in tools] == ["web_search", *NOTION_TOOL_NAMES]


def test_notion_tool_schemas_are_closed_read_only_functions() -> None:
    """Every Notion tool should have a closed schema and no write operation."""
    tools = make_notion_tools()

    assert {tool["name"] for tool in tools} == set(NOTION_TOOL_NAMES)
    for tool in tools:
        assert tool["type"] == "function"
        assert tool["parameters"]["additionalProperties"] is False
        assert "create" not in str(tool["name"])
        assert "append" not in str(tool["name"])


def test_search_returns_normalized_results_and_cursor() -> None:
    """Notion search should return stable page and data source fields."""
    response = Mock(status_code=200)
    response.json.return_value = {
        "results": [
            {
                "object": "page",
                "id": "page-1",
                "url": "https://notion.so/page-1",
                "last_edited_time": "2026-07-31T08:00:00.000Z",
                "properties": {
                    "Name": {
                        "type": "title",
                        "title": [{"plain_text": "Release notes"}],
                    }
                },
            },
            {
                "object": "data_source",
                "id": "source-1",
                "url": "https://notion.so/source-1",
                "last_edited_time": "2026-07-30T08:00:00.000Z",
                "title": [{"plain_text": "Projects"}],
            },
        ],
        "has_more": True,
        "next_cursor": "next-page",
    }
    with patch(
        "flwr.supercore.task_process.connector.notion.requests.request",
        return_value=response,
    ) as request:
        result = search(
            " release ",
            limit=2,
            cursor="opaque-cursor",
            credentials=_CREDENTIALS,
            config={},
            usage_recorder=Mock(),
        )

    request.assert_called_once_with(
        "POST",
        "https://api.notion.com/v1/search",
        headers={
            "Authorization": "Bearer ntn-secret",
            "Content-Type": "application/json",
            "Notion-Version": "2026-03-11",
        },
        json={
            "query": "release",
            "page_size": 2,
            "start_cursor": "opaque-cursor",
        },
        params=None,
        timeout=30.0,
    )
    assert result == {
        "results": [
            {
                "id": "page-1",
                "object": "page",
                "title": "Release notes",
                "url": "https://notion.so/page-1",
                "last_edited_time": "2026-07-31T08:00:00.000Z",
            },
            {
                "id": "source-1",
                "object": "data_source",
                "title": "Projects",
                "url": "https://notion.so/source-1",
                "last_edited_time": "2026-07-30T08:00:00.000Z",
            },
        ],
        "has_more": True,
        "next_cursor": "next-page",
    }


def test_get_page_content_reads_nested_and_paginated_blocks() -> None:
    """Page content should preserve depth-first order across API pages."""
    first_page = Mock(status_code=200)
    first_page.json.return_value = {
        "results": [
            {
                "object": "block",
                "id": "paragraph-1",
                "type": "paragraph",
                "has_children": False,
                "paragraph": {"rich_text": [{"plain_text": "Introduction"}]},
            },
            {
                "object": "block",
                "id": "toggle-1",
                "type": "toggle",
                "has_children": True,
                "toggle": {"rich_text": [{"plain_text": "Details"}]},
            },
        ],
        "has_more": True,
        "next_cursor": "top-page-2",
    }
    nested_page = Mock(status_code=200)
    nested_page.json.return_value = {
        "results": [
            {
                "object": "block",
                "id": "paragraph-2",
                "type": "paragraph",
                "has_children": False,
                "paragraph": {"rich_text": [{"plain_text": "Nested text"}]},
            }
        ],
        "has_more": False,
        "next_cursor": None,
    }
    second_page = Mock(status_code=200)
    second_page.json.return_value = {
        "results": [
            {
                "object": "block",
                "id": "row-1",
                "type": "table_row",
                "has_children": False,
                "table_row": {
                    "cells": [
                        [{"plain_text": "Name"}],
                        [{"plain_text": "Status"}],
                    ]
                },
            }
        ],
        "has_more": False,
        "next_cursor": None,
    }
    with patch(
        "flwr.supercore.task_process.connector.notion.requests.request",
        side_effect=[first_page, nested_page, second_page],
    ) as request:
        result = get_page_content(
            " page-1 ",
            max_blocks=5,
            credentials=_CREDENTIALS,
            config={},
            usage_recorder=Mock(),
        )

    assert [call.args[1] for call in request.call_args_list] == [
        "https://api.notion.com/v1/blocks/page-1/children",
        "https://api.notion.com/v1/blocks/toggle-1/children",
        "https://api.notion.com/v1/blocks/page-1/children",
    ]
    assert [call.kwargs["params"] for call in request.call_args_list] == [
        {"page_size": "5"},
        {"page_size": "3"},
        {"page_size": "2", "start_cursor": "top-page-2"},
    ]
    assert result == {
        "page_id": "page-1",
        "blocks": [
            {
                "id": "paragraph-1",
                "type": "paragraph",
                "text": "Introduction",
                "depth": 0,
                "has_children": False,
            },
            {
                "id": "toggle-1",
                "type": "toggle",
                "text": "Details",
                "depth": 0,
                "has_children": True,
            },
            {
                "id": "paragraph-2",
                "type": "paragraph",
                "text": "Nested text",
                "depth": 1,
                "has_children": False,
            },
            {
                "id": "row-1",
                "type": "table_row",
                "text": "Name | Status",
                "depth": 0,
                "has_children": False,
            },
        ],
        "truncated": False,
    }


def test_get_page_content_stops_before_fetching_children_at_limit() -> None:
    """The block limit should prevent additional nested API requests."""
    response = Mock(status_code=200)
    response.json.return_value = {
        "results": [
            {
                "id": "toggle-1",
                "type": "toggle",
                "has_children": True,
                "toggle": {"rich_text": [{"plain_text": "Details"}]},
            }
        ],
        "has_more": False,
        "next_cursor": None,
    }
    with patch(
        "flwr.supercore.task_process.connector.notion.requests.request",
        return_value=response,
    ) as request:
        result = get_page_content(
            "page-1",
            max_blocks=1,
            credentials=_CREDENTIALS,
            config={},
            usage_recorder=Mock(),
        )

    assert len(result["blocks"]) == 1
    assert result["truncated"] is True
    request.assert_called_once()


@pytest.mark.parametrize("tool_name", NOTION_TOOL_NAMES)
def test_registry_maps_every_notion_tool_to_one_connection(tool_name: str) -> None:
    """All Notion tools should resolve the account's single Notion credential."""
    assert registry.requires_connector_credentials(tool_name)
    assert registry.get_connector_ref(tool_name) == NOTION_CONNECTOR_REF


def test_notion_api_errors_are_stable_and_secret_safe() -> None:
    """Notion response details and bearer tokens must not leak through errors."""
    response = Mock(status_code=401)
    response.json.return_value = {
        "code": "unauthorized",
        "message": "Bearer ntn-secret is invalid",
    }
    with (
        patch(
            "flwr.supercore.task_process.connector.notion.requests.request",
            return_value=response,
        ),
        pytest.raises(NotionApiError) as error,
    ):
        search(
            "release",
            credentials=_CREDENTIALS,
            config={},
            usage_recorder=Mock(),
        )

    assert error.value.code == "unauthorized"
    assert "ntn-secret" not in str(error.value)


@pytest.mark.parametrize(
    ("response", "side_effect", "expected_code"),
    [
        (None, requests.RequestException("ntn-secret"), "request_failed"),
        (Mock(status_code=429), None, "rate_limited"),
        (
            Mock(status_code=500, **{"json.side_effect": ValueError()}),
            None,
            "http_error",
        ),
        (
            Mock(status_code=200, **{"json.side_effect": ValueError()}),
            None,
            "invalid_response",
        ),
    ],
)
def test_notion_transport_failures_are_mapped_without_details(
    response: Mock | None,
    side_effect: Exception | None,
    expected_code: str,
) -> None:
    """Transport, HTTP, and decoding failures should use stable error codes."""
    with (
        patch(
            "flwr.supercore.task_process.connector.notion.requests.request",
            return_value=response,
            side_effect=side_effect,
        ),
        pytest.raises(NotionApiError) as error,
    ):
        search(
            "release",
            credentials=_CREDENTIALS,
            config={},
            usage_recorder=Mock(),
        )

    assert error.value.code == expected_code
    assert "ntn-secret" not in str(error.value)


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (search, {"query": ""}),
        (search, {"query": "valid", "limit": 101}),
        (search, {"query": "valid", "cursor": ""}),
        (get_page_content, {"page_id": ""}),
        (get_page_content, {"page_id": "page-1", "max_blocks": 201}),
    ],
)
def test_notion_tool_inputs_are_validated(
    function: Callable[..., JSONObject], arguments: dict[str, object]
) -> None:
    """Invalid tool inputs should fail before making an HTTP request."""
    with (
        patch(
            "flwr.supercore.task_process.connector.notion.requests.request"
        ) as request,
        pytest.raises(ValueError),
    ):
        function(
            **arguments,
            credentials=_CREDENTIALS,
            config={},
            usage_recorder=Mock(),
        )

    request.assert_not_called()
