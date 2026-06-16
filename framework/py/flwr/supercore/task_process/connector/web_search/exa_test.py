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
"""Tests for the Exa web-search provider."""

from unittest.mock import Mock

import pytest

from flwr.supercore.typing import JSONObject

from .exa import EXA_SEARCH_URL, REQUEST_TIMEOUT, ExaWebSearchProvider, _parse_results


def test_search_calls_exa_and_returns_parsed_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Search requests should call Exa and return normalized results."""
    monkeypatch.setenv("EXA_API_KEY", "exa_test_key")
    payload: JSONObject = {
        "results": [
            {
                "title": " Flower ",
                "url": " https://flower.ai ",
                "summary": " Federated AI framework ",
                "publishedDate": " 2026-06-10T00:00:00Z ",
            }
        ]
    }
    response = Mock(status_code=200, text="")
    response.json.return_value = payload
    post_mock = Mock(return_value=response)
    monkeypatch.setattr(
        "flwr.supercore.task_process.connector.web_search.exa.requests.post",
        post_mock,
    )

    result = ExaWebSearchProvider().search(" flower federated learning ")

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
    post_mock.assert_called_once_with(
        EXA_SEARCH_URL,
        headers={
            "Content-Type": "application/json",
            "x-api-key": "exa_test_key",
        },
        json={"query": "flower federated learning"},
        timeout=REQUEST_TIMEOUT,
    )


def test_parse_results_normalizes_exa_results() -> None:
    """Parser should keep usable Exa results and skip malformed entries."""
    payload: JSONObject = {
        "results": [
            {
                "title": " Flower Docs ",
                "url": " https://flower.ai/docs ",
                "summary": " ",
                "publishedDate": " ",
            },
            "not an object",
            {"title": "Missing URL"},
            {"url": "https://example.com/missing-title"},
            {"title": " ", "url": "https://example.com/blank-title"},
            {"title": "Blank URL", "url": " "},
        ]
    }

    assert _parse_results(payload) == [
        {
            "title": "Flower Docs",
            "url": "https://flower.ai/docs",
            "snippet": None,
            "published_at": None,
        }
    ]
