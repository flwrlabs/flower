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
"""Browser Use Core-backed browser automation adapter."""

from __future__ import annotations

import asyncio
import importlib
import os
from collections.abc import Sequence
from typing import Any, cast

from flwr.supercore.typing import JSONObject, JSONValue

BROWSER_USE_PROVIDER = "browser_use"
BROWSER_USE_MODEL_ENV = "FLWR_BROWSER_USE_MODEL"
BROWSER_USE_LLM_PROVIDER_ENV = "FLWR_BROWSER_USE_LLM_PROVIDER"
BROWSER_USE_OLLAMA_HOST_ENV = "FLWR_BROWSER_USE_OLLAMA_HOST"
BROWSER_USE_ALLOWED_DOMAINS_ENV = "FLWR_BROWSER_USE_ALLOWED_DOMAINS"
BROWSER_USE_HEADLESS_ENV = "FLWR_BROWSER_USE_HEADLESS"
DEFAULT_BROWSER_USE_LLM_PROVIDER = "ollama"
DEFAULT_BROWSER_USE_MODEL = "llama3.1:8b"
_SUPPORTED_LLM_PROVIDERS = frozenset({"browser_use", "ollama"})


class BrowserUseProvider:
    """Browser Use Core adapter."""

    def __init__(
        self,
        *,
        model: str | None = None,
        llm_provider: str | None = None,
        ollama_host: str | None = None,
        headless: bool | None = None,
    ) -> None:
        """Initialize the Browser Use provider."""
        self._llm_provider = _resolve_llm_provider(llm_provider)
        self._model = _resolve_model(model)
        self._ollama_host = _resolve_ollama_host(ollama_host)
        self._headless = _resolve_headless_from_env() if headless is None else headless

    def run(
        self,
        task: str,
        *,
        allowed_domains: Sequence[str] | None = None,
    ) -> JSONObject:
        """Execute one Browser Use Core task."""
        task = task.strip()
        if not task:
            raise ValueError("browser-use requires a non-empty task.")

        domains = _resolve_allowed_domains(allowed_domains)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self._run_async(task=task, allowed_domains=domains),
            )

        raise RuntimeError(
            "Browser Use provider cannot be invoked from a running event loop."
        )

    async def _run_async(
        self,
        *,
        task: str,
        allowed_domains: list[str] | None,
    ) -> JSONObject:
        """Execute one Browser Use Core task asynchronously."""
        agent_type, browser_profile_type, llm_type = _load_browser_use(
            self._llm_provider
        )
        browser_profile = browser_profile_type(
            headless=self._headless,
            allowed_domains=allowed_domains,
        )
        agent = agent_type(
            task=task,
            llm=self._make_llm(llm_type),
            browser_profile=browser_profile,
        )

        try:
            history = await agent.run()
        except Exception as exc:
            raise RuntimeError(f"{BROWSER_USE_PROVIDER} request failed: {exc}") from exc

        return {
            "object": "browser_use.response",
            "status": "completed",
            "output": _final_result(history),
            "metadata": {
                "provider": BROWSER_USE_PROVIDER,
                "llm_provider": self._llm_provider,
                "model": self._model,
                "ollama_host": self._ollama_host,
                "headless": self._headless,
                "allowed_domains": cast(JSONValue, allowed_domains),
            },
        }

    def _make_llm(self, llm_type: type[Any]) -> object:
        """Create the configured Browser Use LLM adapter."""
        if self._llm_provider == "ollama":
            return llm_type(model=self._model, host=self._ollama_host)
        return llm_type(model=self._model)


def _load_browser_use(llm_provider: str) -> tuple[type[Any], type[Any], type[Any]]:
    """Load Browser Use Core classes lazily."""
    try:
        browser_use_beta = importlib.import_module("browser_use.beta")
    except ImportError as exc:
        raise RuntimeError(
            "Install 'browser-use[core]' to use the Browser Use provider."
        ) from exc

    llm_class_name = "ChatOllama"
    if llm_provider == "browser_use":
        llm_class_name = "ChatBrowserUse"

    try:
        return (
            cast(type[Any], browser_use_beta.Agent),
            cast(type[Any], browser_use_beta.BrowserProfile),
            cast(type[Any], getattr(browser_use_beta, llm_class_name)),
        )
    except AttributeError as exc:
        raise RuntimeError(
            "browser-use[core] must expose Agent, BrowserProfile, and the "
            f"{llm_class_name} LLM adapter from browser_use.beta."
        ) from exc


def _resolve_llm_provider(llm_provider: str | None) -> str:
    """Resolve the Browser Use LLM provider."""
    if llm_provider is None:
        llm_provider = os.getenv(BROWSER_USE_LLM_PROVIDER_ENV, "")
    llm_provider = llm_provider.strip().lower().replace("-", "_")
    if not llm_provider:
        llm_provider = DEFAULT_BROWSER_USE_LLM_PROVIDER
    if llm_provider not in _SUPPORTED_LLM_PROVIDERS:
        raise ValueError(
            f"Browser Use LLM provider must be one of "
            f"{sorted(_SUPPORTED_LLM_PROVIDERS)}."
        )
    return llm_provider


def _resolve_model(model: str | None) -> str:
    """Resolve the Browser Use model name."""
    if model is not None:
        model = model.strip()
        if not model:
            raise ValueError("Browser Use model must not be empty.")
        return model

    env_model = os.getenv(BROWSER_USE_MODEL_ENV, "").strip()
    return env_model or DEFAULT_BROWSER_USE_MODEL


def _resolve_ollama_host(ollama_host: str | None) -> str | None:
    """Resolve the Ollama host."""
    if ollama_host is None:
        ollama_host = os.getenv(BROWSER_USE_OLLAMA_HOST_ENV, "")
    ollama_host = ollama_host.strip()
    return ollama_host or None


def _resolve_headless_from_env() -> bool:
    """Resolve headless browser setting from the environment."""
    value = os.getenv(BROWSER_USE_HEADLESS_ENV, "").strip().lower()
    if not value:
        return True
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Environment variable {BROWSER_USE_HEADLESS_ENV} must be a boolean."
    )


def _resolve_allowed_domains(
    allowed_domains: Sequence[str] | None,
) -> list[str] | None:
    """Resolve allowed domains from arguments or environment."""
    if allowed_domains is not None:
        domains = [domain.strip() for domain in allowed_domains if domain.strip()]
        return domains or None

    raw_domains = os.getenv(BROWSER_USE_ALLOWED_DOMAINS_ENV, "")
    domains = [domain.strip() for domain in raw_domains.split(",") if domain.strip()]
    return domains or None


def _final_result(history: object) -> str:
    """Extract the final Browser Use result."""
    final_result = getattr(history, "final_result", None)
    if callable(final_result):
        result = final_result()
    else:
        result = getattr(history, "output", None)

    if result is None:
        return ""
    if isinstance(result, str):
        return result
    return str(result)
