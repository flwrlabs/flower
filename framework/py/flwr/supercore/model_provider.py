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
"""Asynchronous Open Responses client used by Runtime inference."""

from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from time import monotonic
from typing import cast

import httpx
from anyio import CancelScope

from flwr.supercore.tracing import TraceSpan, trace_span
from flwr.supercore.typing import JSONObject, JSONValue

DEFAULT_MODEL_API_ENDPOINT = "https://api.flower.ai/v1/responses"
DEFAULT_MODEL_API_TIMEOUT = 600.0
_TERMINAL_EVENTS = ("response.completed", "response.failed", "response.incomplete")
_SSE_LINE = re.compile(rb"\r\n|\n|\r(?!$)")


class ModelProviderError(RuntimeError):
    """Error returned by the configured model provider."""

    def __init__(self, detail: JSONValue, status_code: int | None = None) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(
            detail
            if isinstance(detail, str)
            else json.dumps(detail, separators=(",", ":"))
        )


def model_api_timeout() -> float:
    """Return the configured provider request timeout."""
    try:
        return max(1.0, float(os.getenv("FLWR_MODEL_API_TIMEOUT", "600").strip()))
    except ValueError:
        return DEFAULT_MODEL_API_TIMEOUT


def model_api_config() -> tuple[str, dict[str, str]]:
    """Return the configured Responses URL and its server-side credentials."""
    key = os.getenv("FLWR_MODEL_API_KEY", "").strip()
    url = os.getenv("FLWR_MODEL_API_ENDPOINT", "").strip().rstrip("/")
    if not url:
        url = DEFAULT_MODEL_API_ENDPOINT
    if not url.endswith("/responses"):
        raise RuntimeError("FLWR_MODEL_API_ENDPOINT must include the /responses path.")
    if not key and url == DEFAULT_MODEL_API_ENDPOINT:
        raise RuntimeError("Model API key is not set (FLWR_MODEL_API_KEY).")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return url, headers


def validate_model_request(payload: JSONObject) -> None:
    """Validate the supported Responses request shape."""
    if not isinstance(payload.get("model"), str) or not payload["model"]:
        raise ValueError("Request requires a non-empty string field 'model'.")
    value = payload.get("input")
    if not isinstance(value, str) and (
        not isinstance(value, list) or not all(isinstance(item, dict) for item in value)
    ):
        raise ValueError("Request field 'input' must be a string or JSON object array.")
    fields = {
        "stream": bool,
        "reasoning": dict,
        "previous_response_id": str,
        "instructions": str,
        "max_output_tokens": int,
        "metadata": dict,
        "text": dict,
    }
    for name, expected in fields.items():
        if name in payload and not isinstance(payload[name], expected):
            raise ValueError(f"Request field '{name}' has an invalid type.")
    if "tools" in payload and (
        not isinstance(payload["tools"], list)
        or not all(isinstance(item, dict) for item in payload["tools"])
    ):
        raise ValueError("Request field 'tools' must be a JSON object array.")


def validate_model_response(payload: object) -> JSONObject:
    """Validate a JSON response or a terminal event's nested response."""
    if not isinstance(payload, dict) or payload.get("object") != "response":
        raise ValueError("Model response field 'object' must be 'response'.")
    for name in ("id", "status"):
        if name in payload and not isinstance(payload[name], str):
            raise ValueError(f"Model response field '{name}' must be a string.")
    if "output" in payload and (
        not isinstance(payload["output"], list)
        or not all(isinstance(item, dict) for item in payload["output"])
    ):
        raise ValueError("Model response field 'output' must be a JSON object array.")
    if payload.get("error") is not None and not isinstance(payload["error"], dict):
        raise ValueError("Model response field 'error' must be a JSON object.")
    return cast(JSONObject, payload)


@asynccontextmanager
async def model_provider_response(
    payload: JSONObject,
) -> AsyncGenerator[httpx.Response, None]:
    """Open the upstream response and close it on completion or cancellation."""
    url, headers = model_api_config()
    async with httpx.AsyncClient(
        timeout=model_api_timeout(), follow_redirects=True
    ) as client:
        with trace_span("model.provider_http"):
            response = await client.send(
                client.build_request("POST", url, headers=headers, json=payload),
                stream=True,
            )
        try:
            if response.status_code >= 400:
                await response.aread()
                try:
                    detail = cast(JSONValue, response.json())
                except ValueError:
                    detail = response.text
                raise ModelProviderError(detail, response.status_code)
            if payload.get("stream") is True and "text/event-stream" not in (
                response.headers.get("content-type", "").lower()
            ):
                raise ModelProviderError("Expected a text/event-stream response.")
            yield response
        finally:
            with CancelScope(shield=True):
                await response.aclose()


async def invoke_model_provider(payload: JSONObject) -> JSONObject:
    """Read one non-streaming response for internal callers such as Browser Use."""
    with trace_span(
        "model.provider", attributes={"flwr.model.name": str(payload["model"])}
    ):
        async with model_provider_response(payload) as response:
            await response.aread()
            return validate_model_response(response.json())


def _observed_event(data: list[bytes], name: str) -> JSONObject:
    """Inspect accounting and timing metadata without validating the relayed stream."""
    if name == "response.output_text.delta":
        return {"type": name}
    if name and name not in _TERMINAL_EVENTS:
        return {}
    try:
        event = json.loads(b"\n".join(data))
    except (ValueError, UnicodeDecodeError):
        return {}
    if not isinstance(event, dict):
        return {}
    event.setdefault("type", name)
    return cast(JSONObject, event)


async def stream_model_provider_response(
    response: httpx.Response,
    record_usage: Callable[[JSONObject], Awaitable[None]],
    span: TraceSpan,
    started: float,
) -> AsyncGenerator[bytes, None]:
    """Forward upstream bytes unchanged, observing only usage and first text."""
    buffer, name = b"", ""
    data: list[bytes] = []
    first_text, usage_recorded = False, False

    async def observe() -> None:
        nonlocal first_text, usage_recorded
        event = _observed_event(data, name)
        if not first_text and event.get("type") == "response.output_text.delta":
            first_text = True
            span.set_attribute(
                "flwr.model.first_text_ms", (monotonic() - started) * 1000
            )
            span.add_event("provider.first_text")
        body = event.get("response")
        if (
            not usage_recorded
            and event.get("type") in _TERMINAL_EVENTS
            and isinstance(body, dict)
        ):
            await record_usage(body)
            usage_recorded = True

    async for chunk in response.aiter_bytes():
        buffer += chunk
        while match := _SSE_LINE.search(buffer):
            line, buffer = buffer[: match.start()], buffer[match.end() :]
            if not line:
                await observe()
                data, name = [], ""
            elif line.startswith(b"data:"):
                value = line[5:]
                data.append(value[1:] if value.startswith(b" ") else value)
            elif line.startswith(b"event:"):
                name = line[6:].strip().decode("utf-8", errors="replace")
        yield chunk
    # Some providers omit the final blank separator.
    if buffer.startswith(b"data:"):
        data.append(buffer[5:].lstrip(b" "))
    if data:
        await observe()
