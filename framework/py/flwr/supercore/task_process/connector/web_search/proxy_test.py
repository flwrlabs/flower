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
"""Tests for the proxy-backed web search provider."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import Mock

import pytest
import requests

from flwr.supercore.typing import JSONObject

from .proxy import ProxyWebSearchProvider

_PROXY_ENDPOINT = "http://proxy/v1/web-search"


@dataclass
class _Response:
    status_code: int = 200
    body: object | None = None
    text: str = ""

    def json(self) -> object:
        """Return the mocked JSON response body."""
        return self.body


def test_proxy_web_search_provider_posts_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proxy provider should post the query and return the proxy payload."""
    payload: JSONObject = {"results": []}
    post_mock = Mock(return_value=_Response(body=payload))
    monkeypatch.setattr(requests, "post", post_mock)

    result = ProxyWebSearchProvider(_PROXY_ENDPOINT).search(" Flower ")

    assert result == payload
    post_mock.assert_called_once_with(
        _PROXY_ENDPOINT,
        json={"query": "Flower"},
        timeout=60.0,
    )


@pytest.mark.parametrize(
    ("payload", "expected_message"),
    [
        (["not an object"], "invalid JSON"),
        ({}, "results list"),
        ({"results": "not a list"}, "results list"),
    ],
)
def test_proxy_web_search_provider_rejects_invalid_top_level_shapes(
    monkeypatch: pytest.MonkeyPatch, payload: object, expected_message: str
) -> None:
    """Proxy responses should be validated only at the top level."""
    monkeypatch.setattr(
        requests,
        "post",
        Mock(return_value=_Response(body=payload)),
    )

    with pytest.raises(RuntimeError, match=expected_message):
        ProxyWebSearchProvider(_PROXY_ENDPOINT).search("Flower")
