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
"""Tests for the built-in web search connector."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import Mock

import pytest
import requests

from . import WEB_SEARCH_ENDPOINT_ENV, search
from .brave import BRAVE_API_KEY_ENV, BRAVE_WEB_SEARCH_URL

_PROXY_ENDPOINT = "http://proxy/v1/web-search"


@dataclass
class _Response:
    status_code: int = 200
    body: object | None = None
    text: str = ""

    def json(self) -> object:
        """Return the mocked JSON response body."""
        return self.body


def test_search_calls_proxy_endpoint_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proxy mode should route search requests through the proxy provider."""
    monkeypatch.setenv(WEB_SEARCH_ENDPOINT_ENV, _PROXY_ENDPOINT)
    provider = Mock()
    provider.search.return_value = {"results": []}
    provider_cls = Mock(return_value=provider)
    monkeypatch.setattr(
        "flwr.supercore.task_process.connector.web_search.ProxyWebSearchProvider",
        provider_cls,
    )

    assert search("Flower") == {"results": []}

    provider_cls.assert_called_once_with(_PROXY_ENDPOINT)
    provider.search.assert_called_once_with("Flower")


def test_search_proxy_takes_precedence_over_direct_provider_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proxy mode should not inspect or fall back to direct provider keys."""
    monkeypatch.setenv(WEB_SEARCH_ENDPOINT_ENV, _PROXY_ENDPOINT)
    monkeypatch.setenv(BRAVE_API_KEY_ENV, "brave_test_key")
    provider = Mock()
    provider.search.side_effect = RuntimeError("proxy unavailable")
    provider_cls = Mock(return_value=provider)
    get_mock = Mock()
    monkeypatch.setattr(
        "flwr.supercore.task_process.connector.web_search.ProxyWebSearchProvider",
        provider_cls,
    )
    monkeypatch.setattr(requests, "get", get_mock)

    with pytest.raises(RuntimeError, match="proxy unavailable"):
        search("Flower")

    provider_cls.assert_called_once_with(_PROXY_ENDPOINT)
    provider.search.assert_called_once_with("Flower")
    get_mock.assert_not_called()


def test_search_uses_direct_providers_when_proxy_endpoint_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct provider behavior should remain available without proxy config."""
    monkeypatch.setenv(BRAVE_API_KEY_ENV, "brave_test_key")
    response = _Response(body={"web": {"results": []}})
    get_mock = Mock(return_value=response)
    post_mock = Mock()
    monkeypatch.setattr(requests, "get", get_mock)
    monkeypatch.setattr(requests, "post", post_mock)

    assert search("Flower") == {"results": []}
    get_mock.assert_called_once()
    assert get_mock.call_args.args == (BRAVE_WEB_SEARCH_URL,)
    post_mock.assert_not_called()
