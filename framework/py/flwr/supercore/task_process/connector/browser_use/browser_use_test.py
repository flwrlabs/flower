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
"""Tests for the Browser Use connector."""


from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest

from .browser_use import FlowerResponsesChatModel, invoke_browser_use_provider


class _History:
    """Fake Browser Use history."""

    def final_result(self) -> str:
        """Return the final Browser Use result."""
        return "Final answer"


def test_invoke_browser_use_provider_uses_flower_headless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Browser Use should run headless with Flower's Responses-backed adapter."""
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

    class _Agent:
        def __init__(
            self,
            *,
            task: str,
            llm: FlowerResponsesChatModel,
            browser_profile: _BrowserProfile,
            enable_signal_handler: bool,
        ) -> None:
            self.task = task
            self.llm = llm
            self.browser_profile = browser_profile
            self.enable_signal_handler = enable_signal_handler
            created["agent"] = self
            created["llm"] = llm

        async def run(self) -> _History:
            """Return fake agent history."""
            return _History()

    monkeypatch.setattr(
        "flwr.supercore.task_process.connector.browser_use.browser_use.Agent",
        _Agent,
    )
    monkeypatch.setattr(
        "flwr.supercore.task_process.connector.browser_use.browser_use.BrowserProfile",
        _BrowserProfile,
    )

    result = invoke_browser_use_provider(
        " Find Flower docs ",
        allowed_domains=["*.flower.ai", "docs.python.org"],
        model=" gpt-5 ",
    )

    assert result == {
        "object": "browser_use.response",
        "status": "completed",
        "output": "Final answer",
        "metadata": {
            "llm_provider": "flower",
            "model": "gpt-5",
            "headless": True,
            "allowed_domains": ["*.flower.ai", "docs.python.org"],
        },
    }
    agent = created["agent"]
    assert isinstance(agent, _Agent)
    assert agent.task == "Find Flower docs"
    assert not agent.enable_signal_handler

    llm = created["llm"]
    assert isinstance(llm, FlowerResponsesChatModel)
    assert llm.model == "gpt-5"


@pytest.mark.parametrize(
    "provider_response",
    [
        {
            "object": "response",
            "output_text": "Click the first link.",
        },
        {
            "object": "response",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Click the first link.",
                        }
                    ],
                }
            ],
        },
    ],
)
def test_flower_responses_chat_model_invokes_model_provider(
    monkeypatch: pytest.MonkeyPatch,
    provider_response: dict[str, object],
) -> None:
    """FlowerResponsesChatModel should send Browser Use messages to Responses."""

    class _ChatInvokeCompletion:
        """Fake Browser Use chat completion result."""

        def __init__(
            self,
            *,
            completion: object,
            usage: object | None,
            stop_reason: str | None,
        ) -> None:
            self.completion = completion
            self.usage = usage
            self.stop_reason = stop_reason

    invoke_model_provider = Mock(return_value=provider_response)
    monkeypatch.setattr(
        "flwr.supercore.task_process.connector.browser_use.browser_use.invoke_model_provider",
        invoke_model_provider,
    )
    browser_use_module = ModuleType("browser_use")
    llm_module = ModuleType("browser_use.llm")
    views_module = ModuleType("browser_use.llm.views")
    browser_use_module.__dict__["__path__"] = []
    browser_use_module.__dict__["llm"] = llm_module
    llm_module.__dict__["__path__"] = []
    llm_module.__dict__["views"] = views_module
    views_module.__dict__["ChatInvokeCompletion"] = _ChatInvokeCompletion
    monkeypatch.setitem(sys.modules, "browser_use", browser_use_module)
    monkeypatch.setitem(sys.modules, "browser_use.llm", llm_module)
    monkeypatch.setitem(sys.modules, "browser_use.llm.views", views_module)

    messages: list[Any] = [
        SimpleNamespace(role="system", text="You control a browser."),
        SimpleNamespace(role="user", text="Open example.com."),
    ]

    result = cast(
        Any,
        asyncio.run(FlowerResponsesChatModel(model="gpt-5").ainvoke(messages)),
    )

    assert result.completion == "Click the first link."
    invoke_model_provider.assert_called_once_with(
        {
            "model": "gpt-5",
            "input": [
                {"role": "system", "content": "You control a browser."},
                {"role": "user", "content": "Open example.com."},
            ],
            "stream": False,
        }
    )
