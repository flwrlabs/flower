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

from . import registry
from .notion import (
    NOTION_CONNECTOR_REF,
    NOTION_TOOL_NAMES,
    NotionApiError,
    get_page_content,
    search,
)

_CREDENTIALS = {"access_token": "ntn-secret"}


def _list_response(
    results: list[dict[str, object]],
    *,
    next_cursor: str | None = None,
    status_code: int = 200,
) -> Mock:
    response = Mock(status_code=status_code)
    response.json.return_value = {
        "results": results,
        "has_more": next_cursor is not None,
        "next_cursor": next_cursor,
    }
    return response


def _text_block(
    block_id: str, block_type: str, text: str, *, has_children: bool = False
) -> dict[str, object]:
    return {
        "id": block_id,
        "type": block_type,
        "has_children": has_children,
        block_type: {"rich_text": [{"plain_text": text}]},
    }


def test_notion_tools_are_registered_as_read_only_credentials() -> None:
    """Notion tools should be closed schemas backed by one OAuth connection."""
    tools = registry.get_connector_tools(NOTION_CONNECTOR_REF)

    assert [tool["name"] for tool in tools] == list(NOTION_TOOL_NAMES)
    assert not registry.has_builtin_connector(NOTION_CONNECTOR_REF)
    for tool in tools:
        name = str(tool["name"])
        assert tool["type"] == "function"
        assert tool["parameters"]["additionalProperties"] is False
        assert "create" not in name and "append" not in name
        assert registry.requires_connector_credentials(name)
        assert registry.get_connector_ref(name) == NOTION_CONNECTOR_REF


def test_search_calls_notion_and_normalizes_results() -> None:
    """Search should preserve cursors and normalize page and data source titles."""
    response = _list_response(
        [
            {
                "object": "page",
                "id": "page-1",
                "url": "https://notion.so/page-1",
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
                "title": [{"plain_text": "Projects"}],
            },
        ],
        next_cursor="next-page",
    )
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

    assert request.call_args.args == ("POST", "https://api.notion.com/v1/search")
    assert request.call_args.kwargs["json"] == {
        "query": "release",
        "page_size": 2,
        "start_cursor": "opaque-cursor",
    }
    assert request.call_args.kwargs["headers"]["Notion-Version"] == "2026-03-11"
    rows = result["results"]
    assert isinstance(rows, list)
    assert [(row["object"], row["title"]) for row in rows] == [
        ("page", "Release notes"),
        ("data_source", "Projects"),
    ]
    assert result["next_cursor"] == "next-page"


def test_get_page_content_reads_nested_and_paginated_blocks() -> None:
    """Page content should preserve depth-first order across API pages."""
    first_page = _list_response(
        [
            _text_block("paragraph-1", "paragraph", "Introduction"),
            _text_block("toggle-1", "toggle", "Details", has_children=True),
        ],
        next_cursor="top-page-2",
    )
    nested_page = _list_response(
        [_text_block("paragraph-2", "paragraph", "Nested text")]
    )
    second_page = _list_response(
        [
            {
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
        ]
    )
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
    assert request.call_args_list[-1].kwargs["params"]["start_cursor"] == ("top-page-2")
    blocks = result["blocks"]
    assert isinstance(blocks, list)
    assert [(block["id"], block["text"], block["depth"]) for block in blocks] == [
        ("paragraph-1", "Introduction", 0),
        ("toggle-1", "Details", 0),
        ("paragraph-2", "Nested text", 1),
        ("row-1", "Name | Status", 0),
    ]
    assert result["truncated"] is False


def test_get_page_content_stops_at_block_limit() -> None:
    """The block limit should prevent additional nested API requests."""
    response = _list_response(
        [_text_block("toggle-1", "toggle", "Details", has_children=True)]
    )
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

    assert result["truncated"] is True
    request.assert_called_once()


_UNAUTHORIZED = Mock(status_code=401)
_UNAUTHORIZED.json.return_value = {
    "code": "unauthorized",
    "message": "Bearer ntn-secret is invalid",
}


@pytest.mark.parametrize(
    ("response", "side_effect", "expected_code"),
    [
        (_UNAUTHORIZED, None, "unauthorized"),
        (Mock(status_code=429), None, "rate_limited"),
        (None, requests.RequestException("ntn-secret"), "request_failed"),
        (
            Mock(status_code=200, **{"json.side_effect": ValueError()}),
            None,
            "invalid_response",
        ),
    ],
)
def test_api_failures_are_stable_and_secret_safe(
    response: Mock | None,
    side_effect: Exception | None,
    expected_code: str,
) -> None:
    """Provider and transport failures should expose only stable error codes."""
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
        (search, {"query": "valid", "cursor": ""}),
        (get_page_content, {"page_id": "page-1", "max_blocks": 201}),
    ],
)
def test_invalid_inputs_fail_before_request(
    function: Callable[..., object], arguments: dict[str, object]
) -> None:
    """Representative invalid inputs should fail before any API request."""
    with (
        patch("flwr.supercore.task_process.connector.notion.requests.request") as call,
        pytest.raises(ValueError),
    ):
        function(
            **arguments,
            credentials=_CREDENTIALS,
            config={},
            usage_recorder=Mock(),
        )

    call.assert_not_called()
