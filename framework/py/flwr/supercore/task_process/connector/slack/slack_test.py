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
