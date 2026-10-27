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
from collections.abc import AsyncGenerator, Callable, Iterator
from itertools import chain
from time import monotonic
from typing import cast

import httpx
import requests
from anyio import CancelScope

from flwr.supercore.task_process.usage import (
    TaskUsageRecorder,
    task_usage_from_open_response,
)
from flwr.supercore.tracing import trace_span
from flwr.supercore.typing import JSONObject, JSONValue

DEFAULT_MODEL_API_ENDPOINT = "https://api.flower.ai/v1/responses"
DEFAULT_MODEL_API_TIMEOUT = 600.0
_DEFAULT_PROVIDER = "unknown"
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


def model_api_timeout() -> float:
    """Return the configured model provider request timeout."""
    raw_timeout = os.getenv(
        "FLWR_MODEL_API_TIMEOUT",
        str(DEFAULT_MODEL_API_TIMEOUT),
    )
    try:
        timeout = float(raw_timeout.strip())
    except ValueError:
        timeout = DEFAULT_MODEL_API_TIMEOUT
    return max(1.0, timeout)


def model_api_config() -> tuple[str, dict[str, str]]:
    """Resolve the same endpoint and credentials for workers and direct calls.

    Returns
    -------
    tuple[str, dict[str, str]]
        The configured Responses URL and HTTP request headers.
    """
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

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return responses_url, headers


def invoke_model_provider(
    request: JSONObject,
    *,
    usage_recorder: TaskUsageRecorder,
    on_stream_event: Callable[[JSONObject], None] | None = None,
) -> JSONObject:
    """Invoke the configured Open Responses-compatible model provider.

    Control flow:
    1. Read API key, endpoint, and timeout settings from the environment.
    2. Copy the request payload to avoid mutating the caller's object.
    3. Send the request through one provider path that handles normal and
       streaming responses.
    """
    responses_url, headers = model_api_config()
    with trace_span("model.provider", attributes={"flwr.component": "model"}) as span:
        started_at = monotonic()
        if isinstance(request.get("model"), str):
            span.set_attribute("flwr.model.name", cast(str, request["model"]))
        first_text = False
        first_event = False

        def forward_event(event: JSONObject) -> None:
            nonlocal first_text, first_event
            if not first_event:
                first_event = True
                span.add_event("provider.first_event")
            if not first_text and event.get("type") == "response.output_text.delta":
                first_text = True
                span.set_attribute(
                    "flwr.model.first_text_ms", (monotonic() - started_at) * 1000
                )
                span.add_event("provider.first_text")
            if on_stream_event is not None:
                on_stream_event(event)

        response = _invoke_provider_response(
            responses_url=responses_url,
            headers=headers,
            timeout=model_api_timeout(),
            request=dict(request),
            on_stream_event=forward_event,
            usage_recorder=usage_recorder,
            provider=_provider_from_request(request),
        )
        usage = response.get("usage")
        if isinstance(usage, dict):
            for key in ("input_tokens", "output_tokens"):
                count = usage.get(key)
                if (
                    isinstance(count, int)
                    and not isinstance(count, bool)
                    and count >= 0
                ):
                    span.set_attribute(f"flwr.model.{key}", count)
        return response


def _invoke_provider_response(  # pylint: disable=too-many-locals,too-many-branches,too-many-arguments,too-many-statements
    *,
    responses_url: str,
    headers: dict[str, str],
    timeout: float,
    request: JSONObject,
    on_stream_event: Callable[[JSONObject], None] | None,
    usage_recorder: TaskUsageRecorder,
    provider: str,
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
        # For streams this ends at response headers, not at the first SSE event.
        with trace_span("model.provider_http"):
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
        response_payload = _ensure_json_object(payload)
        _record_model_usage(response_payload, usage_recorder, provider)
        return response_payload

    # Streaming parsing only works for Server-Sent Event responses.
    content_type = response.headers.get("Content-Type", "").lower()
    if _STREAM_CONTENT_TYPE not in content_type:
        raise ModelProviderError(
            status_code=response.status_code,
            detail=f"Expected streaming response Content-Type "
            f"{_STREAM_CONTENT_TYPE}, got {content_type or '<missing>'}.",
        )

    last_event: JSONObject | None = None
    for event in _iter_sse_events(response):
        last_event = event
        if on_stream_event is not None:
            on_stream_event(event)

        # Terminal failure events stop the stream and surface the provider event.
        event_type = event.get("type")
        is_failure_event = (
            isinstance(event_type, str) and event_type in _TERMINAL_FAILURE_EVENTS
        )

        if is_failure_event:
            raw_response = event.get("response")
            if isinstance(raw_response, dict):
                _record_model_usage(raw_response, usage_recorder, provider)
            else:
                _record_model_usage(event, usage_recorder, provider)
            raise ModelProviderError(
                status_code=response.status_code,
                detail=event,
            )

        # Terminal success events carry the final response object.
        if isinstance(event_type, str) and event_type in _TERMINAL_SUCCESS_EVENTS:
            raw_response = event.get("response")
            if isinstance(raw_response, dict):
                final_response = raw_response
                _record_model_usage(final_response, usage_recorder, provider)
                return final_response
            _record_model_usage(event, usage_recorder, provider)
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


def _record_model_usage(
    response: JSONObject, usage_recorder: TaskUsageRecorder, provider: str
) -> None:
    usage = task_usage_from_open_response(response, provider=provider)
    if usage is not None:
        usage_recorder.record(usage)


def _provider_from_request(request: JSONObject) -> str:
    """Return the provider identifier for a model provider request."""
    model = request.get("model")
    if isinstance(model, str) and model:
        return model
    return _DEFAULT_PROVIDER


class _ModelEventDecoder:
    """Decode provider SSE fields and JSON identically on both transports."""

    def __init__(self) -> None:
        self.event_name: str | None = None
        self.data: list[str] = []

    def feed(self, line: str) -> JSONObject | None:
        """Consume a line, or flush pending data with an empty line at EOF."""
        if line.startswith("event:"):
            self.event_name = line.removeprefix("event:").strip() or None
        elif line.startswith("data:"):
            data = line.removeprefix("data:")
            self.data.append(data[1:] if data.startswith(" ") else data)
        elif not line and self.data:
            encoded = "\n".join(self.data)
            name, self.event_name = self.event_name, None
            self.data.clear()
            if encoded.strip() == "[DONE]":
                return None
            try:
                event = _ensure_json_object(json.loads(encoded))
            except json.JSONDecodeError as exc:
                raise ModelProviderError(
                    detail=encoded,
                    message="Model provider stream returned invalid JSON event",
                ) from exc
            if name is not None and not isinstance(event.get("type"), str):
                event["type"] = name
            return event
        return None


def _iter_sse_events(response: requests.Response) -> Iterator[JSONObject]:
    decoder = _ModelEventDecoder()
    for raw_line in chain(response.iter_lines(), [b""]):
        event = decoder.feed(raw_line.decode("utf-8"))
        if event is not None:
            yield event


async def _aiter_sse_events(
    response: httpx.Response,
) -> AsyncGenerator[JSONObject, None]:
    decoder = _ModelEventDecoder()
    async for line in response.aiter_lines():
        event = decoder.feed(line)
        if event is not None:
            yield event
    event = decoder.feed("")
    if event is not None:
        yield event


async def iter_model_provider_response(
    request: JSONObject,
) -> AsyncGenerator[JSONObject, None]:
    """Yield JSON or SSE objects from the configured provider without blocking.

    Parameters
    ----------
    request : JSONObject
        A validated Responses request, including its streaming preference.

    Yields
    ------
    JSONObject
        The response body, or individual SSE events through a terminal event.
    """
    url, headers = model_api_config()
    async with httpx.AsyncClient(
        timeout=model_api_timeout(), follow_redirects=True
    ) as client:
        with trace_span("model.provider_http"):
            response = await client.send(
                client.build_request("POST", url, headers=headers, json=request),
                stream=True,
            )
        try:
            if response.status_code >= 400:
                await response.aread()
                try:
                    detail = cast(JSONValue, response.json())
                except ValueError:
                    detail = response.text
                raise ModelProviderError(
                    detail=detail, status_code=response.status_code
                )
            if request.get("stream") is not True:
                await response.aread()
                yield _ensure_json_object(response.json())
                return
            if (
                _STREAM_CONTENT_TYPE
                not in response.headers.get("content-type", "").lower()
            ):
                raise ModelProviderError(detail="Expected a text/event-stream response")
            async for event in _aiter_sse_events(response):
                yield event
                if event.get("type") in (
                    _TERMINAL_SUCCESS_EVENTS | _TERMINAL_FAILURE_EVENTS
                ):
                    return
            raise ModelProviderError(
                detail="Model stream ended before a terminal event"
            )
        finally:
            with CancelScope(shield=True):
                await response.aclose()
