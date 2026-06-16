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
"""Tests for the Brave web-search provider."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from flwr.supercore.typing import JSONObject

from .brave import (
    BRAVE_WEB_SEARCH_URL,
    REQUEST_TIMEOUT,
    BraveWebSearchProvider,
    _parse_results,
)


def test_search_calls_brave_and_returns_parsed_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Search requests should call Brave and return normalized results."""
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave_test_key")
    payload: JSONObject = {
        "web": {
            "results": [
                {
                    "title": " Flower ",
                    "url": " https://flower.ai ",
                    "description": " Federated AI framework ",
                    "page_age": " 2026-06-10T00:00:00Z ",
                }
            ]
        }
    }
    response = Mock(status_code=200, text="")
    response.json.return_value = payload
    get_mock = Mock(return_value=response)
    monkeypatch.setattr(
        "flwr.supercore.task_process.connector.web_search.brave.requests.get",
        get_mock,
    )

    result = BraveWebSearchProvider().search(" flower federated learning ")

    assert result == {
        "results": [
            {
                "title": "Flower",
                "url": "https://flower.ai",
                "snippet": "Federated AI framework",
                "published_at": "2026-06-10T00:00:00Z",
            }
        ]
    }
    get_mock.assert_called_once_with(
        BRAVE_WEB_SEARCH_URL,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": "brave_test_key",
        },
        params={"q": "flower federated learning"},
        timeout=REQUEST_TIMEOUT,
    )


def test_parse_results_normalizes_brave_results() -> None:
    """Parser should keep usable Brave results and skip malformed entries."""
    payload: JSONObject = {
        "web": {
            "results": [
                {
                    "title": " Flower Docs ",
                    "url": " https://flower.ai/docs ",
                    "description": " ",
                    "page_age": " ",
                },
                "not an object",
                {"title": "Missing URL"},
                {"url": "https://example.com/missing-title"},
                {"title": " ", "url": "https://example.com/blank-title"},
                {"title": "Blank URL", "url": " "},
            ]
        }
    }

    assert _parse_results(payload) == [
        {
            "title": "Flower Docs",
            "url": "https://flower.ai/docs",
            "snippet": None,
            "published_at": None,
        }
    ]
