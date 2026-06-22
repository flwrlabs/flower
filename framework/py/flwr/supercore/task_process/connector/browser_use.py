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
from typing import TYPE_CHECKING, Any, cast

from flwr.supercore.task_process.model.provider import invoke_model_provider
from flwr.supercore.typing import JSONObject, JSONValue

if TYPE_CHECKING:
    from browser_use.agent.views import AgentHistoryList
    from browser_use.browser.profile import BrowserProfile
    from browser_use.llm.base import BaseChatModel
    from browser_use.llm.messages import BaseMessage
    from browser_use.llm.views import ChatInvokeCompletion
    from pydantic import BaseModel
else:

    class BaseChatModel:
        """Runtime fallback for Browser Use's optional BaseChatModel protocol."""


BROWSER_USE_CONNECTOR_NAME = "browser_use"
_DEFAULT_BROWSER_USE_MODEL = "flwrlabs/lizzy-long-context"
_LLM_PROVIDER = "flower"
_HEADLESS = True


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
        allowed_domains: list[str] | None = None,
    ) -> JSONObject:
        """Execute one Browser Use task."""
        task = task.strip()
        if not task:
            raise ValueError("browser_use requires a non-empty task.")

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # Connector handlers are synchronous, while Browser Use runs async.
            return asyncio.run(
                self._run_async(task=task, allowed_domains=allowed_domains)
            )

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
            from browser_use import (  # pylint: disable=import-outside-toplevel
                Agent,
                BrowserProfile,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Install 'browser-use[core]' to use the browser_use connector."
            ) from exc

        # Browser Use drives the browser and calls this chat adapter for each step.
        browser_profile: BrowserProfile = BrowserProfile(
            headless=_HEADLESS,
            allowed_domains=allowed_domains,
        )
        agent: Agent[Any, Any] = Agent(
            task=task,
            llm=FlowerResponsesChatModel(model=self._model),
            browser_profile=browser_profile,
        )

        try:
            history: AgentHistoryList[Any] = await agent.run()
        except Exception as exc:
            raise RuntimeError(f"browser_use request failed: {exc}") from exc

        result = history.final_result()
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


class FlowerResponsesChatModel(BaseChatModel):
    """Browser Use LLM adapter backed by Flower's Responses API."""

    def __init__(self, *, model: str) -> None:
        """Initialize the Flower Responses chat model."""
        self.model: str = model

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

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        output_format: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> ChatInvokeCompletion[Any]:
        """Invoke Flower's Responses API with Browser Use messages."""
        del kwargs
        request: JSONObject = {
            "model": self.model,
            "input": [
                {"role": message.role, "content": message.text} for message in messages
            ],
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
                schema = output_format.model_json_schema()
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
        # Some Responses-compatible providers include the SDK-style convenience
        # field directly on the response object.
        raw_output_text = response.get("output_text")
        if isinstance(raw_output_text, str):
            output_text = raw_output_text
        else:
            # Raw Open Responses payloads keep message text in output content
            # items, so collect those text chunks when output_text is absent.
            content_texts: list[str] = []
            output = response.get("output")
            if isinstance(output, list):
                for output_item in output:
                    if not isinstance(output_item, dict):
                        continue
                    content = output_item.get("content")
                    if not isinstance(content, list):
                        continue
                    for content_item in content:
                        if not isinstance(content_item, dict):
                            continue
                        text = content_item.get("text")
                        if isinstance(text, str):
                            content_texts.append(text)
            if not content_texts:
                raise RuntimeError(
                    "Model provider response did not include assistant output text."
                )
            output_text = "".join(content_texts)
        from browser_use.llm.views import (  # pylint: disable=import-outside-toplevel
            ChatInvokeCompletion,
        )

        if output_format is not None:
            return ChatInvokeCompletion(
                completion=output_format.model_validate_json(output_text),
                usage=None,
                stop_reason=None,
            )
        return ChatInvokeCompletion(
            completion=output_text,
            usage=None,
            stop_reason=None,
        )


def invoke_browser_use_provider(
    task: str,
    allowed_domains: list[str] | None = None,
    model: str | None = None,
) -> JSONObject:
    """Execute one Browser Use connector request."""
    return BrowserUseProvider(model=model).invoke(task, allowed_domains=allowed_domains)


__all__ = [
    "BROWSER_USE_CONNECTOR_NAME",
    "invoke_browser_use_provider",
    "make_browser_use_tool",
]
