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
"""Browser Use-backed browser automation connector."""

from __future__ import annotations

import asyncio
import importlib
import json
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, TypeVar, cast

from flwr.supercore.task_process.model.provider import invoke_model_provider
from flwr.supercore.typing import JSONObject, JSONValue

BROWSER_USE_CONNECTOR_NAME = "browser_use"
_DEFAULT_BROWSER_USE_MODEL = "flwrlabs/lizzy-long-context"
_LLM_PROVIDER = "flower"
_HEADLESS = True

T = TypeVar("T")


def make_browser_use_tool() -> JSONObject:
    """Return the browser use function tool schema."""
    return {
        "type": "function",
        "name": BROWSER_USE_CONNECTOR_NAME,
        "description": "Use a headless browser to complete a web task.",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The browser task to complete.",
                },
            },
            "required": ["task"],
            "additionalProperties": False,
        },
    }


class BrowserUseProvider:
    """Browser Use Core adapter backed by Flower Responses."""

    def __init__(
        self,
        *,
        model: str | None = None,
    ) -> None:
        """Initialize the Browser Use provider."""
        self._model = _DEFAULT_BROWSER_USE_MODEL
        if model is not None:
            self._model = model.strip()
            if not self._model:
                raise ValueError("Browser Use model must not be empty.")

    def invoke(
        self,
        task: str,
        *,
        allowed_domains: Sequence[str] | None = None,
    ) -> JSONObject:
        """Execute one Browser Use task."""
        task = task.strip()
        if not task:
            raise ValueError("browser_use requires a non-empty task.")

        domains: list[str] | None = None
        if allowed_domains is not None:
            if isinstance(allowed_domains, str):
                raise ValueError("allowed_domains must be a list of strings.")
            domains = []
            for domain in allowed_domains:
                if not isinstance(domain, str):
                    raise ValueError("allowed_domains must contain only strings.")
                domain = domain.strip()
                if domain:
                    domains.append(domain)
            if not domains:
                domains = None

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # Connector handlers are synchronous, while Browser Use runs async.
            return asyncio.run(self._run_async(task=task, allowed_domains=domains))

        raise RuntimeError("browser_use cannot run inside an active event loop.")

    async def _run_async(
        self,
        *,
        task: str,
        allowed_domains: list[str] | None,
    ) -> JSONObject:
        """Execute one Browser Use task asynchronously."""
        try:
            # Browser Use is optional, so import it only when the connector runs.
            browser_use = importlib.import_module("browser_use")
            agent_type = cast(type[Any], browser_use.Agent)
            browser_profile_type = cast(type[Any], browser_use.BrowserProfile)
        except ImportError as exc:
            raise RuntimeError(
                "Install 'browser-use[core]' to use the browser_use connector."
            ) from exc
        except AttributeError as exc:
            raise RuntimeError(
                "browser-use[core] must expose Agent and BrowserProfile."
            ) from exc

        # Browser Use drives the browser and calls this chat adapter for each step.
        agent = agent_type(
            task=task,
            llm=FlowerResponsesChatModel(model=self._model),
            browser_profile=browser_profile_type(
                headless=_HEADLESS,
                allowed_domains=allowed_domains,
            ),
        )

        try:
            history = await agent.run()
        except Exception as exc:
            raise RuntimeError(f"browser_use request failed: {exc}") from exc

        final_result = getattr(history, "final_result", None)
        result = (
            final_result() if callable(final_result) else getattr(history, "output", "")
        )
        if result is None:
            result = ""

        return {
            "object": "browser_use.response",
            "status": "completed",
            "output": result if isinstance(result, str) else str(result),
            "metadata": {
                "llm_provider": _LLM_PROVIDER,
                "model": self._model,
                "headless": _HEADLESS,
                "allowed_domains": cast(JSONValue, allowed_domains),
            },
        }


class FlowerResponsesChatModel:
    """Browser Use LLM adapter backed by Flower's Responses API."""

    _verified_api_keys = False

    def __init__(self, *, model: str) -> None:
        """Initialize the Flower Responses chat model."""
        self.model = model

    @property
    def provider(self) -> str:
        """Return the provider name."""
        return "flower"

    @property
    def name(self) -> str:
        """Return the model name."""
        return self.model

    @property
    def model_name(self) -> str:
        """Return the model name for legacy Browser Use callers."""
        return self.model

    async def ainvoke(  # pylint: disable=too-many-branches,too-many-locals,too-many-nested-blocks,too-many-statements
        self,
        messages: list[object],
        output_format: type[T] | None = None,
        **kwargs: Any,
    ) -> object:
        """Invoke Flower's Responses API with Browser Use messages."""
        del kwargs
        input_messages: list[JSONObject] = []
        for message in messages:
            # Browser Use passes chat-like message objects, not raw Responses input.
            role_value = getattr(message, "role", "user")
            role = (
                role_value
                if isinstance(role_value, str)
                and role_value in {"user", "system", "assistant"}
                else "user"
            )
            text = getattr(message, "text", None)
            if not isinstance(text, str):
                content = getattr(message, "content", None)
                if isinstance(content, str):
                    text = content
                elif isinstance(content, Sequence) and not isinstance(content, str):
                    parts = []
                    for part in content:
                        part_text = getattr(part, "text", None)
                        if isinstance(part_text, str):
                            parts.append(part_text)
                    text = "\n".join(parts)
                else:
                    text = str(message)
            input_messages.append({"role": role, "content": text})

        request: JSONObject = {
            "model": self.model,
            "input": input_messages,
            "stream": False,
        }
        if output_format is not None:
            try:
                # Reuse Browser Use's schema optimizer for structured step outputs.
                schema_module = importlib.import_module("browser_use.llm.schema")
                schema_optimizer = schema_module.SchemaOptimizer
                schema = schema_optimizer.create_optimized_json_schema(
                    output_format,
                    remove_min_items=True,
                    remove_defaults=True,
                )
            except (AttributeError, ImportError):
                schema = cast(Any, output_format).model_json_schema()
            if not isinstance(schema, dict):
                raise TypeError("Browser Use output schema must be a JSON object.")
            request["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "agent_output",
                    "strict": True,
                    "schema": cast(JSONObject, schema),
                }
            }

        response = await asyncio.to_thread(invoke_model_provider, request)
        output_text = response.get("output_text")
        if not isinstance(output_text, str):
            output = response.get("output")
            text_parts = []
            if isinstance(output, Sequence) and not isinstance(output, str):
                for item in output:
                    if not isinstance(item, dict):
                        continue
                    content = item.get("content")
                    if isinstance(content, Sequence) and not isinstance(content, str):
                        for content_item in content:
                            if not isinstance(content_item, dict):
                                continue
                            text = content_item.get("text")
                            if isinstance(text, str):
                                text_parts.append(text)
                    text = item.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
            output_text = (
                "\n".join(text_parts)
                if text_parts
                else json.dumps(response, separators=(",", ":"))
            )

        if output_format is not None:
            completion: object = cast(Any, output_format).model_validate_json(
                output_text
            )
        else:
            completion = output_text

        try:
            # Browser Use expects ChatInvokeCompletion when its views module exists.
            views = importlib.import_module("browser_use.llm.views")
            completion_type = views.ChatInvokeCompletion
            return completion_type(completion=completion, usage=None, stop_reason=None)
        except ImportError:
            return SimpleNamespace(completion=completion, usage=None, stop_reason=None)


def invoke_browser_use_provider(
    task: str,
    allowed_domains: Sequence[str] | None = None,
    model: str | None = None,
) -> JSONObject:
    """Execute one Browser Use connector request."""
    return BrowserUseProvider(model=model).invoke(task, allowed_domains=allowed_domains)


__all__ = [
    "BROWSER_USE_CONNECTOR_NAME",
    "invoke_browser_use_provider",
    "make_browser_use_tool",
]
