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
"""Open Responses-compatible provider client for Model task processes."""


from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from typing import cast

import requests

from flwr.supercore.typing import JSONObject, JSONValue

DEFAULT_MODEL_API_ENDPOINT = "https://api.flower.ai/v1/responses"
DEFAULT_MODEL_API_TIMEOUT = 180.0
_STREAM_CONTENT_TYPE = "text/event-stream"
_TERMINAL_SUCCESS_EVENTS = frozenset({"response.completed", "response.incomplete"})
_TERMINAL_FAILURE_EVENTS = frozenset({"error", "response.failed"})


class ModelProviderError(RuntimeError):
    """Error returned by the configured model provider."""

    def __init__(
        self,
        *,
        detail: JSONValue,
        status_code: int | None = None,
        message: str = "Model provider request failed",
    ) -> None:
        """Initialize the provider error."""
        self.status_code = status_code
        self.detail = detail
        if isinstance(detail, str):
            formatted_detail = detail
        else:
            formatted_detail = json.dumps(detail, separators=(",", ":"))
        if status_code is None:
            super().__init__(f"{message}: {formatted_detail}")
        else:
            super().__init__(f"{message}: {status_code} {formatted_detail}")


def invoke_model_provider(
    request: JSONObject,
    *,
    on_stream_event: Callable[[JSONObject], None] | None = None,
) -> JSONObject:
    """Invoke the configured Open Responses-compatible model provider.

    Control flow:
    1. Read API key, endpoint, and timeout settings from the environment.
    2. Copy the request payload to avoid mutating the caller's object.
    3. Send the request through one provider path that handles normal and
       streaming responses.
    """
    # Resolve provider configuration from environment variables.
    api_key = os.getenv("FLWR_MODEL_API_KEY", "").strip()
    responses_url = os.getenv("FLWR_MODEL_API_ENDPOINT", "").strip()
    if not responses_url:
        if not api_key:
            raise RuntimeError("Model API key is not set (FLWR_MODEL_API_KEY).")
        responses_url = DEFAULT_MODEL_API_ENDPOINT
    responses_url = responses_url.rstrip("/")
    if not responses_url.endswith("/responses"):
        raise RuntimeError(
            "Model API endpoint must include the /responses path "
            "(FLWR_MODEL_API_ENDPOINT)."
        )
    if not api_key and responses_url == DEFAULT_MODEL_API_ENDPOINT:
        raise RuntimeError("Model API key is not set (FLWR_MODEL_API_KEY).")

    raw_timeout = os.getenv(
        "FLWR_MODEL_API_TIMEOUT",
        str(DEFAULT_MODEL_API_TIMEOUT),
    )
    try:
        timeout = float(raw_timeout.strip())
    except ValueError:
        timeout = DEFAULT_MODEL_API_TIMEOUT
    timeout = max(1.0, timeout)

    # Build request metadata once, then execute the shared provider path.
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = dict(request)
    return _invoke_provider_response(
        responses_url=responses_url,
        headers=headers,
        timeout=timeout,
        request=payload,
        on_stream_event=on_stream_event,
    )


def _invoke_provider_response(  # pylint: disable=too-many-locals,too-many-branches
    *,
    responses_url: str,
    headers: dict[str, str],
    timeout: float,
    request: JSONObject,
    on_stream_event: Callable[[JSONObject], None] | None,
) -> JSONObject:
    """Run a normal or streaming provider request.

    Control flow:
    1. Detect whether the caller requested streaming.
    2. POST the request and fail immediately for HTTP error status codes.
    3. For normal responses, parse and return the response JSON object.
    4. For streaming responses, validate SSE content and consume events until a
       terminal success or failure event arrives.
    """
    stream = request.get("stream") is True

    # Send one HTTP request and let HTTP status represent transport failure.
    try:
        response = requests.post(
            responses_url,
            headers=headers,
            json=request,
            timeout=timeout,
            stream=stream,
        )
    except requests.RequestException as exc:
        raise ModelProviderError(detail=str(exc)) from exc

    if response.status_code >= 400:
        try:
            detail = cast(JSONValue, response.json())
        except ValueError:
            detail = response.text
        raise ModelProviderError(
            status_code=response.status_code,
            detail=detail,
        )

    if not stream:
        # Successful non-streaming responses must be JSON objects.
        try:
            payload = response.json()
        except ValueError as exc:
            raise ModelProviderError(
                detail=response.text or "no response body",
                message="Model provider returned invalid JSON",
            ) from exc
        return _ensure_json_object(payload)

    # Streaming parsing only works for Server-Sent Event responses.
    content_type = response.headers.get("Content-Type", "").lower()
    if _STREAM_CONTENT_TYPE not in content_type:
        raise ModelProviderError(
            status_code=response.status_code,
            detail=f"Expected streaming response Content-Type "
            f"{_STREAM_CONTENT_TYPE}, got {content_type or '<missing>'}.",
        )

    last_event: JSONObject | None = None
    for event_name, data in _iter_sse_events(response):
        if data.strip() == "[DONE]":
            continue

        # Each SSE data payload is an Open Responses stream event object.
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ModelProviderError(
                detail=data,
                message="Model provider stream returned invalid JSON event",
            ) from exc
        event = _ensure_json_object(payload)

        # Some providers put the event type in the SSE event name instead of JSON.
        if event_name is not None and not isinstance(event.get("type"), str):
            event = dict(event)
            event["type"] = event_name

        last_event = event
        if on_stream_event is not None:
            on_stream_event(event)

        # Terminal failure events stop the stream and surface the provider event.
        event_type = event.get("type")
        is_failure_event = (
            isinstance(event_type, str) and event_type in _TERMINAL_FAILURE_EVENTS
        )

        if is_failure_event:
            raise ModelProviderError(
                status_code=response.status_code,
                detail=event,
            )

        # Terminal success events carry the final response object.
        if isinstance(event_type, str) and event_type in _TERMINAL_SUCCESS_EVENTS:
            response_payload = event.get("response")
            if isinstance(response_payload, dict):
                return cast(JSONObject, response_payload)
            return event

    raise ModelProviderError(
        status_code=response.status_code,
        detail=last_event if last_event is not None else "no event was received",
        message="Model provider stream ended before a terminal event",
    )


def _ensure_json_object(payload: object) -> JSONObject:
    if not isinstance(payload, dict):
        detail = cast(JSONValue, payload)
        raise ModelProviderError(
            detail=detail,
            message="Model provider returned a non-object JSON payload",
        )
    return cast(JSONObject, payload)


def _iter_sse_events(response: requests.Response) -> Iterator[tuple[str | None, str]]:
    event_name: str | None = None
    data_lines: list[str] = []

    for raw_line in response.iter_lines():
        line = raw_line.decode("utf-8")

        # A blank line terminates one SSE event.
        if not line:
            if data_lines:
                yield event_name, "\n".join(data_lines)
                event_name = None
                data_lines = []
            continue

        # Lines starting with ":" are SSE comments or keepalives.
        if line.startswith(":"):
            continue

        # The event name is optional; data lines carry the event payload.
        if line.startswith("event:"):
            event_name = line.removeprefix("event:").strip() or None
            continue
        if line.startswith("data:"):
            data = line.removeprefix("data:")

            # SSE allows one optional space after the field separator.
            if data.startswith(" "):
                data = data[1:]
            data_lines.append(data)

    if data_lines:
        yield event_name, "\n".join(data_lines)
