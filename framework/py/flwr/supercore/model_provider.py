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
from collections.abc import AsyncGenerator
from time import monotonic
from typing import cast

import httpx
from anyio import CancelScope

from flwr.supercore.tracing import trace_span
from flwr.supercore.typing import JSONObject, JSONValue

DEFAULT_MODEL_API_ENDPOINT = "https://api.flower.ai/v1/responses"
DEFAULT_MODEL_API_TIMEOUT = 600.0
TERMINAL_EVENTS = frozenset(
    {"error", "response.completed", "response.failed", "response.incomplete"}
)


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


def _decode_event(data: list[str], name: str | None) -> JSONObject | None:
    encoded = "\n".join(data)
    if encoded.strip() == "[DONE]":
        return None
    event = json.loads(encoded)
    if not isinstance(event, dict):
        raise ValueError("Model provider returned a non-object event.")
    if name and not isinstance(event.get("type"), str):
        event["type"] = name
    event_type = event.get("type")
    if not isinstance(event_type, str) or "\n" in event_type or "\r" in event_type:
        raise ValueError("Model provider returned an invalid event type.")
    if event_type in {"response.completed", "response.incomplete"}:
        validate_model_response(event.get("response"))
    return cast(JSONObject, event)


async def _stream_events(response: httpx.Response) -> AsyncGenerator[JSONObject, None]:
    data: list[str] = []
    name: str | None = None
    async for line in response.aiter_lines():
        if line.startswith("event:"):
            name = line[6:].strip() or None
        elif line.startswith("data:"):
            data.append(line[6:] if line.startswith("data: ") else line[5:])
        elif not line:
            if data:
                event = _decode_event(data, name)
                if event is not None:
                    yield event
            data, name = [], None
    if data:
        event = _decode_event(data, name)
        if event is not None:
            yield event


async def iter_model_provider_response(
    payload: JSONObject,
) -> AsyncGenerator[JSONObject, None]:
    """Yield the JSON response or validated provider SSE events through completion."""
    url, headers = model_api_config()
    attributes = {"flwr.model.name": cast(str, payload["model"])}
    with trace_span("model.provider", attributes=attributes) as span:
        started, first_text = monotonic(), False
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
                if payload.get("stream") is not True:
                    await response.aread()
                    yield validate_model_response(response.json())
                    return
                if (
                    "text/event-stream"
                    not in response.headers.get("content-type", "").lower()
                ):
                    raise ModelProviderError("Expected a text/event-stream response.")
                async for event in _stream_events(response):
                    if not first_text and event["type"] == "response.output_text.delta":
                        first_text = True
                        span.set_attribute(
                            "flwr.model.first_text_ms", (monotonic() - started) * 1000
                        )
                        span.add_event("provider.first_text")
                    yield event
                    if event["type"] in TERMINAL_EVENTS:
                        return
                raise ModelProviderError("Model stream ended before a terminal event.")
            except GeneratorExit:
                # Callers close the iterator after consuming a JSON response.
                return
            finally:
                with CancelScope(shield=True):
                    await response.aclose()
