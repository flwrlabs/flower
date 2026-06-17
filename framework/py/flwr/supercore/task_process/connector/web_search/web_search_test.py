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
"""Tests for built-in websearch connector configuration."""

from unittest.mock import Mock

import pytest

from flwr.supercore.task_process.connector import web_search

from .brave import BRAVE_API_KEY
from .exa import EXA_API_KEY
from .tavily import TAVILY_API_KEY


def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear all websearch provider API key env vars."""
    for env_var in (
        BRAVE_API_KEY,
        TAVILY_API_KEY,
        EXA_API_KEY,
    ):
        monkeypatch.delenv(env_var, raising=False)


def test_search_fails_fast_without_existing_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Websearch should require one existing provider API key env var."""
    _clear_provider_env(monkeypatch)
    brave_provider = Mock()
    tavily_provider = Mock()
    exa_provider = Mock()
    monkeypatch.setattr(web_search, "BraveWebSearchProvider", brave_provider)
    monkeypatch.setattr(web_search, "TavilyWebSearchProvider", tavily_provider)
    monkeypatch.setattr(web_search, "ExaWebSearchProvider", exa_provider)

    with pytest.raises(RuntimeError, match=BRAVE_API_KEY):
        web_search.search("Flower")

    brave_provider.assert_not_called()
    tavily_provider.assert_not_called()
    exa_provider.assert_not_called()


def test_search_accepts_existing_brave_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Websearch should accept the existing Brave env var."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv(BRAVE_API_KEY, "brave_test_key")
    provider_instance = Mock()
    provider_instance.search.return_value = {"results": []}
    provider = Mock(return_value=provider_instance)
    monkeypatch.setattr(web_search, "BraveWebSearchProvider", provider)

    assert web_search.search("Flower") == {"results": []}

    provider.assert_called_once_with()
    provider_instance.search.assert_called_once_with("Flower")


def test_search_accepts_existing_tavily_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Websearch should accept the existing Tavily env var."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv(TAVILY_API_KEY, "tavily_test_key")
    provider_instance = Mock()
    provider_instance.search.return_value = {"results": []}
    provider = Mock(return_value=provider_instance)
    monkeypatch.setattr(web_search, "TavilyWebSearchProvider", provider)

    assert web_search.search("Flower") == {"results": []}

    provider.assert_called_once_with()
    provider_instance.search.assert_called_once_with("Flower")


def test_search_accepts_existing_exa_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Websearch should accept the existing Exa env var."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv(EXA_API_KEY, "exa_test_key")
    provider_instance = Mock()
    provider_instance.search.return_value = {"results": []}
    provider = Mock(return_value=provider_instance)
    monkeypatch.setattr(web_search, "ExaWebSearchProvider", provider)

    assert web_search.search("Flower") == {"results": []}

    provider.assert_called_once_with()
    provider_instance.search.assert_called_once_with("Flower")
