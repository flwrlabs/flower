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
"""Responses-compatible model provider runner."""


from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import requests

from flwr.supercore.task_message import JsonObject

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
OPENAI_ORG_ID_ENV = "OPENAI_ORG_ID"
OPENAI_PROJECT_ID_ENV = "OPENAI_PROJECT_ID"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_REQUEST_TIMEOUT = 120.0


@dataclass(frozen=True)
class ModelProviderResult:
    """Responses-compatible provider result."""

    response: JsonObject
    events: list[JsonObject]


class ModelProviderError(Exception):
    """Model provider failure with a task-message-compatible error payload."""

    def __init__(self, error: JsonObject) -> None:
        super().__init__(str(error.get("message", "Model provider failed.")))
        self.error = error


def invoke_responses_model(
    request: JsonObject,
    on_stream_event: Callable[[JsonObject], None] | None = None,
) -> ModelProviderResult:
    """Invoke an OpenAI Responses-compatible model provider."""
    stream = request.get("stream", False)
    if not isinstance(stream, bool):
        raise ModelProviderError(
            _error_payload("invalid_request", "`stream` must be a boolean.")
        )

    url = f"{_base_url()}/responses"
    headers = _headers()
    if stream:
        return _invoke_streaming_responses_model(url, headers, request, on_stream_event)
    return _invoke_non_streaming_responses_model(url, headers, request)


def _invoke_non_streaming_responses_model(
    url: str, headers: dict[str, str], request: JsonObject
) -> ModelProviderResult:
    """Invoke a non-streaming Responses-compatible provider."""
    try:
        response = requests.post(
            url,
            headers=headers,
            json=request,
            timeout=DEFAULT_REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise ModelProviderError(_error_payload("request_error", str(exc))) from exc

    _raise_for_http_error(response)
    return ModelProviderResult(response=_decode_response_json(response), events=[])


def _invoke_streaming_responses_model(
    url: str,
    headers: dict[str, str],
    request: JsonObject,
    on_stream_event: Callable[[JsonObject], None] | None,
) -> ModelProviderResult:
    """Invoke a streaming Responses-compatible provider and collect events."""
    try:
        response = requests.post(
            url,
            headers=headers,
            json=request,
            stream=True,
            timeout=DEFAULT_REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise ModelProviderError(_error_payload("request_error", str(exc))) from exc

    try:
        _raise_for_http_error(response)
        events: list[JsonObject] = []
        final_response: JsonObject | None = None
        for line in response.iter_lines(decode_unicode=True):
            event = _decode_sse_line(line)
            if event is None:
                continue
            events.append(event)
            if on_stream_event is not None:
                on_stream_event(event)
            if event.get("type") == "response.completed" and isinstance(
                event.get("response"), dict
            ):
                final_response = cast(JsonObject, event["response"])

        if final_response is None:
            final_response = cast(JsonObject, {"events": events})
        _ensure_json_object(final_response)
        return ModelProviderResult(response=final_response, events=events)
    finally:
        response.close()


def _base_url() -> str:
    """Return the configured provider base URL."""
    return os.environ.get(OPENAI_BASE_URL_ENV, DEFAULT_OPENAI_BASE_URL).rstrip("/")


def _headers() -> dict[str, str]:
    """Return provider request headers."""
    api_key = os.environ.get(OPENAI_API_KEY_ENV)
    if not api_key:
        raise ModelProviderError(
            _error_payload(
                "missing_api_key",
                f"`{OPENAI_API_KEY_ENV}` must be set to invoke the model provider.",
            )
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if org_id := os.environ.get(OPENAI_ORG_ID_ENV):
        headers["OpenAI-Organization"] = org_id
    if project_id := os.environ.get(OPENAI_PROJECT_ID_ENV):
        headers["OpenAI-Project"] = project_id
    return headers


def _raise_for_http_error(response: requests.Response) -> None:
    """Raise a structured provider error for unsuccessful HTTP responses."""
    if response.status_code < 400:
        return
    error = _error_payload(
        "http_error",
        f"Model provider returned HTTP {response.status_code}.",
    )
    error["status_code"] = response.status_code
    try:
        body = response.json()
    except ValueError:
        body = response.text
    if isinstance(body, dict):
        error["response"] = cast(JsonObject, body)
    elif isinstance(body, str):
        error["response_text"] = body
    raise ModelProviderError(error)


def _decode_response_json(response: requests.Response) -> JsonObject:
    """Decode a provider JSON response object."""
    try:
        decoded = response.json()
    except ValueError as exc:
        raise ModelProviderError(
            _error_payload("invalid_response", "Model provider returned invalid JSON.")
        ) from exc
    if not isinstance(decoded, dict):
        raise ModelProviderError(
            _error_payload(
                "invalid_response",
                "Model provider response must be a JSON object.",
            )
        )
    return _ensure_json_object(cast(JsonObject, decoded))


def _decode_sse_line(line: bytes | str) -> JsonObject | None:
    """Decode one Server-Sent Events line into a JSON event object."""
    if isinstance(line, bytes):
        line = line.decode("utf-8")
    line = line.strip()
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None

    data = line.removeprefix("data:").strip()
    if data == "[DONE]":
        return None
    try:
        event = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ModelProviderError(
            _error_payload("invalid_stream_event", "Stream event is invalid JSON.")
        ) from exc
    if not isinstance(event, dict):
        raise ModelProviderError(
            _error_payload(
                "invalid_stream_event",
                "Stream event must be a JSON object.",
            )
        )
    return _ensure_json_object(cast(JsonObject, event))


def _ensure_json_object(payload: JsonObject) -> JsonObject:
    """Validate that a payload is strict JSON-compatible."""
    try:
        json.dumps(payload, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ModelProviderError(
            _error_payload(
                "invalid_json",
                "Model provider payload contains non-JSON values.",
            )
        ) from exc
    return payload


def _error_payload(error_type: str, message: str) -> JsonObject:
    """Create a structured provider error payload."""
    return {"type": error_type, "message": message}
