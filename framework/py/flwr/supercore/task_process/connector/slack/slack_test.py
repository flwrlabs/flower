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
"""Tests for Slack request mapping and API errors."""

from typing import cast
from unittest.mock import Mock, patch

import pytest

from flwr.supercore.typing import JSONObject

from .. import registry
from .executors import SlackApiError

_HTTP_REQUEST = "flwr.supercore.task_process.connector.http.requests.request"
_CREDENTIALS: JSONObject = {"access_token": "xoxp-secret"}


@pytest.mark.parametrize(
    ("tool_name", "channel_types"),
    (
        ("slack_search_public", ["public_channel"]),
        (
            "slack_search_public_and_private",
            ["public_channel", "private_channel", "mpim", "im"],
        ),
    ),
)
def test_search_request(tool_name: str, channel_types: list[str]) -> None:
    """Map search fields and return Slack's response unchanged."""
    payload: JSONObject = {"ok": True, "results": {"messages": []}}
    response = Mock(status_code=200)
    response.json.return_value = payload
    with patch(_HTTP_REQUEST, return_value=response) as request:
        result = registry.invoke_connector(
            tool_name,
            {
                "keywords": ["alpha", "beta"],
                "filters": "in:<#C1>",
                "natural_language_query": "release plan",
                "content_types": "messages,files",
                "after": "1700000000",
                "limit": 21,
                "include_context": False,
            },
            Mock(),
            _CREDENTIALS,
            {},
        )
    assert result is payload
    request.assert_called_once()
    assert request.call_args.args == (
        "POST",
        "https://slack.com/api/assistant.search.context",
    )
    assert request.call_args.kwargs["json"] == {
        "query": "release plan in:<#C1>",
        "term_clauses": ["alpha", "beta"],
        "modifiers": "in:<#C1>",
        "content_types": ["messages", "files"],
        "channel_types": channel_types,
        "after": 1700000000,
        "limit": 21,
        "include_bots": False,
        "include_context_messages": False,
    }


def test_slack_api_error() -> None:
    """Surface Slack's error without retrying through another API."""
    response = Mock(status_code=200)
    response.json.return_value = {"ok": False, "error": "invalid_arguments"}
    with (
        patch(_HTTP_REQUEST, return_value=response) as request,
        pytest.raises(SlackApiError, match="invalid_arguments"),
    ):
        registry.invoke_connector(
            "slack_search_public", {"query": "release"}, Mock(), _CREDENTIALS, {}
        )
    request.assert_called_once()


@pytest.mark.parametrize("response_format", ("detailed", "concise"))
def test_search_options(response_format: str) -> None:
    """Filter joined conversations and honor context length and concise output."""
    pages = [
        {
            "ok": True,
            "results": {
                "messages": [
                    {
                        "channel_id": "C1",
                        "content": "release",
                        "blocks": [],
                        "context_messages": {
                            "before": [{"text": "before text"}],
                            "after": [{"text": "after text"}],
                        },
                    },
                    {"channel_id": "C2", "content": "not joined"},
                ],
                "files": [
                    {"file_id": "F1", "title": "joined file", "file_type": "pdf"},
                    {"file_id": "F2", "title": "not joined"},
                ],
            },
            "response_metadata": {"next_cursor": "search-next"},
        },
        {
            "ok": True,
            "channels": [{"id": "C1"}],
            "response_metadata": {"next_cursor": "membership-next"},
        },
        {"ok": True, "channels": [{"id": "C3"}]},
        {"ok": True, "file": {"channels": ["C3"]}},
        {"ok": True, "file": {"channels": ["C2"]}},
    ]
    with patch(
        _HTTP_REQUEST,
        side_effect=[
            Mock(status_code=200, json=Mock(return_value=page)) for page in pages
        ],
    ) as request:
        result = registry.invoke_connector(
            "slack_search_public",
            {
                "query": "release",
                "content_types": "messages,files",
                "only_my_channels": True,
                "max_context_length": 5,
                "response_format": response_format,
            },
            Mock(),
            _CREDENTIALS,
            {},
        )
    assert request.call_count == 5
    assert request.call_args_list[2].kwargs["params"]["cursor"] == "membership-next"
    assert isinstance(result, dict)
    assert result["response_metadata"] == {"next_cursor": "search-next"}
    results = cast(dict[str, list[JSONObject]], result["results"])
    message = results["messages"][0]
    file = results["files"][0]
    assert len(results["messages"]) == 1
    assert len(results["files"]) == 1
    assert message["channel_id"] == "C1"
    assert file["file_id"] == "F1"
    if response_format == "concise":
        assert message == {"channel_id": "C1", "content": "release"}
        assert file == {"file_id": "F1", "title": "joined file"}
    else:
        assert message["context_messages"] == {
            "before": [{"text": "befor"}],
            "after": [{"text": "after"}],
        }
