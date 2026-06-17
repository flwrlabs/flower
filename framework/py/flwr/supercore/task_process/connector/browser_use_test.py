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
"""Tests for the Browser Use Core provider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from .browser_use import (
    BROWSER_USE_ALLOWED_DOMAINS_ENV,
    BROWSER_USE_HEADLESS_ENV,
    BROWSER_USE_LLM_PROVIDER_ENV,
    BROWSER_USE_MODEL_ENV,
    BROWSER_USE_OLLAMA_HOST_ENV,
    BrowserUseProvider,
)


class _History:
    """Fake Browser Use history."""

    def final_result(self) -> str:
        """Return the final Browser Use result."""
        return "Final answer"


def test_run_calls_browser_use_core_and_returns_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Browser Use requests should call the Core agent and normalize results."""
    created: dict[str, object] = {}

    class _BrowserProfile:
        def __init__(
            self,
            *,
            headless: bool,
            allowed_domains: list[str] | None,
        ) -> None:
            self.headless = headless
            self.allowed_domains = allowed_domains
            created["browser_profile"] = self

    class _ChatOllama:
        def __init__(self, *, model: str, host: str | None) -> None:
            self.model = model
            self.host = host
            created["llm"] = self

    class _Agent:
        def __init__(
            self,
            *,
            task: str,
            llm: _ChatOllama,
            browser_profile: _BrowserProfile,
        ) -> None:
            self.task = task
            self.llm = llm
            self.browser_profile = browser_profile
            created["agent"] = self

        async def run(self) -> _History:
            """Return fake agent history."""
            return _History()

    import_module = Mock(
        return_value=SimpleNamespace(
            Agent=_Agent,
            BrowserProfile=_BrowserProfile,
            ChatOllama=_ChatOllama,
        )
    )
    monkeypatch.setattr(
        "flwr.supercore.task_process.connector.browser_use.importlib.import_module",
        import_module,
    )

    result = BrowserUseProvider(
        model=" llama3.1:8b ",
        headless=True,
        ollama_host=" http://127.0.0.1:11434 ",
    ).run(
        " Find Flower docs ",
        allowed_domains=[" *.flower.ai ", "", "docs.python.org"],
    )

    assert result == {
        "object": "browser_use.response",
        "status": "completed",
        "output": "Final answer",
        "metadata": {
            "provider": "browser_use",
            "llm_provider": "ollama",
            "model": "llama3.1:8b",
            "ollama_host": "http://127.0.0.1:11434",
            "headless": True,
            "allowed_domains": ["*.flower.ai", "docs.python.org"],
        },
    }
    import_module.assert_called_once_with("browser_use.beta")

    browser_profile = created["browser_profile"]
    assert isinstance(browser_profile, _BrowserProfile)
    assert browser_profile.headless is True
    assert browser_profile.allowed_domains == ["*.flower.ai", "docs.python.org"]

    llm = created["llm"]
    assert isinstance(llm, _ChatOllama)
    assert llm.model == "llama3.1:8b"
    assert llm.host == "http://127.0.0.1:11434"

    agent = created["agent"]
    assert isinstance(agent, _Agent)
    assert agent.task == "Find Flower docs"


def test_run_reads_environment_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Browser Use provider should read minimal runtime settings from env vars."""

    class _BrowserProfile:
        def __init__(
            self,
            *,
            headless: bool,
            allowed_domains: list[str] | None,
        ) -> None:
            self.headless = headless
            self.allowed_domains = allowed_domains

    class _ChatOllama:
        def __init__(self, *, model: str, host: str | None) -> None:
            self.model = model
            self.host = host

    class _Agent:
        def __init__(
            self,
            *,
            task: str,
            llm: _ChatOllama,
            browser_profile: _BrowserProfile,
        ) -> None:
            self.task = task
            self.llm = llm
            self.browser_profile = browser_profile

        async def run(self) -> _History:
            """Return fake agent history."""
            return _History()

    import_module = Mock(
        return_value=SimpleNamespace(
            Agent=_Agent,
            BrowserProfile=_BrowserProfile,
            ChatOllama=_ChatOllama,
        )
    )
    monkeypatch.setattr(
        "flwr.supercore.task_process.connector.browser_use.importlib.import_module",
        import_module,
    )
    monkeypatch.setenv(BROWSER_USE_LLM_PROVIDER_ENV, "ollama")
    monkeypatch.setenv(BROWSER_USE_MODEL_ENV, "llama3.1:8b")
    monkeypatch.setenv(BROWSER_USE_OLLAMA_HOST_ENV, "http://localhost:11434")
    monkeypatch.setenv(BROWSER_USE_HEADLESS_ENV, "false")
    monkeypatch.setenv(BROWSER_USE_ALLOWED_DOMAINS_ENV, "*.github.com, flower.ai")

    result = BrowserUseProvider().run("Find repo stars")

    assert result["metadata"] == {
        "provider": "browser_use",
        "llm_provider": "ollama",
        "model": "llama3.1:8b",
        "ollama_host": "http://localhost:11434",
        "headless": False,
        "allowed_domains": ["*.github.com", "flower.ai"],
    }


def test_run_rejects_empty_task() -> None:
    """Browser Use requests should require a non-empty task."""
    with pytest.raises(ValueError, match="non-empty task"):
        BrowserUseProvider().run(" ")


def test_missing_browser_use_dependency_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing Browser Use Core should produce an install-focused error."""
    import_module = Mock(side_effect=ImportError("missing"))
    monkeypatch.setattr(
        "flwr.supercore.task_process.connector.browser_use.importlib.import_module",
        import_module,
    )

    with pytest.raises(RuntimeError, match=r"Install 'browser-use\[core\]'"):
        BrowserUseProvider().run("Find repo stars")


def test_invalid_headless_environment_value_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid boolean env var values should fail before browser launch."""
    monkeypatch.setenv(BROWSER_USE_HEADLESS_ENV, "maybe")

    with pytest.raises(ValueError, match=BROWSER_USE_HEADLESS_ENV):
        BrowserUseProvider()
