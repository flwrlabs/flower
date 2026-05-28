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
"""Private Open Responses-compatible provider client for Model executors."""


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


def invoke_responses_model(
    request: JSONObject,
    *,
    on_stream_event: Callable[[JSONObject], None] | None = None,
) -> JSONObject:
    """Invoke the configured Open Responses-compatible model provider.

    Control flow:
    1. Read API key, endpoint, and timeout settings from the environment.
    2. Copy the request payload to avoid mutating the caller's object.
    3. Route streaming requests to `_invoke_streaming_response`; route all
       other requests to `_invoke_response`.
    """
    api_key = os.getenv("FLWR_MODEL_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Model API key is not set (FLWR_MODEL_API_KEY).")

    responses_url = os.getenv("FLWR_MODEL_API_ENDPOINT", "").strip()
    if not responses_url:
        responses_url = DEFAULT_MODEL_API_ENDPOINT
    responses_url = responses_url.rstrip("/")
    if not responses_url.endswith("/responses"):
        raise RuntimeError(
            "Model API endpoint must include the /responses path "
            "(FLWR_MODEL_API_ENDPOINT)."
        )

    raw_timeout = os.getenv(
        "FLWR_MODEL_API_TIMEOUT",
        str(DEFAULT_MODEL_API_TIMEOUT),
    )
    try:
        timeout = float(raw_timeout.strip())
    except ValueError:
        timeout = DEFAULT_MODEL_API_TIMEOUT
    timeout = max(1.0, timeout)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = dict(request)
    if payload.get("stream") is True:
        return _invoke_streaming_response(
            responses_url=responses_url,
            headers=headers,
            timeout=timeout,
            request=payload,
            on_stream_event=on_stream_event,
        )
    return _invoke_response(
        responses_url=responses_url,
        headers=headers,
        timeout=timeout,
        request=payload,
    )


def _invoke_response(
    *,
    responses_url: str,
    headers: dict[str, str],
    timeout: float,
    request: JSONObject,
) -> JSONObject:
    """Run a non-streaming provider request.

    Control flow:
    1. POST the request with streaming disabled.
    2. Fail immediately for HTTP error status codes.
    3. Parse the response body as a JSON object.
    4. Return the parsed response object.
    """
    response = _post_responses_request(
        responses_url=responses_url,
        headers=headers,
        timeout=timeout,
        request=request,
        stream=False,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            _failure_message(
                status_code=response.status_code,
                detail=_response_detail(response),
            )
        )

    return _parse_json_object(
        parse=response.json,
        invalid_json_message="Model provider returned invalid JSON.",
        non_object_message="Model provider returned a non-object JSON payload",
    )


def _invoke_streaming_response(
    *,
    responses_url: str,
    headers: dict[str, str],
    timeout: float,
    request: JSONObject,
    on_stream_event: Callable[[JSONObject], None] | None,
) -> JSONObject:
    """Run a streaming provider request.

    Control flow:
    1. Force `stream` to true and POST the request with streaming enabled.
    2. Fail immediately for HTTP errors or non-SSE response content.
    3. Parse each SSE data payload as a JSON object and forward it to the
       optional stream callback.
    4. Raise on provider failure events; return the terminal success response.
    """
    request["stream"] = True
    response = _post_responses_request(
        responses_url=responses_url,
        headers=headers,
        timeout=timeout,
        request=request,
        stream=True,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            _failure_message(
                status_code=response.status_code,
                detail=_response_detail(response),
            )
        )

    content_type = response.headers.get("Content-Type", "").lower()
    if _STREAM_CONTENT_TYPE not in content_type:
        raise RuntimeError(
            _failure_message(
                status_code=response.status_code,
                detail=f"Expected streaming response Content-Type "
                f"{_STREAM_CONTENT_TYPE}, got {content_type or '<missing>'}.",
            )
        )

    last_event: JSONObject | None = None
    for event_name, data in _iter_sse_events(response):
        if data.strip() == "[DONE]":
            continue
        event = _parse_json_object(
            parse=lambda data=data: json.loads(data),
            invalid_json_message=f"Model provider stream returned invalid JSON event: "
            f"{data}",
            non_object_message="Model provider stream returned a non-object JSON event",
        )
        if event_name is not None and not isinstance(event.get("type"), str):
            event = dict(event)
            event["type"] = event_name

        last_event = event
        if on_stream_event is not None:
            on_stream_event(event)

        event_type = event.get("type")
        is_failure_event = (
            isinstance(event_type, str) and event_type in _TERMINAL_FAILURE_EVENTS
        )

        if is_failure_event:
            raise RuntimeError(
                _failure_message(
                    status_code=response.status_code,
                    detail=event,
                )
            )

        if isinstance(event_type, str) and event_type in _TERMINAL_SUCCESS_EVENTS:
            response_payload = event.get("response")
            if isinstance(response_payload, dict):
                return cast(JSONObject, response_payload)
            return event

    raise RuntimeError(
        _failure_message(status_code=response.status_code, detail=last_event)
    )


def _post_responses_request(
    *,
    responses_url: str,
    headers: dict[str, str],
    timeout: float,
    request: JSONObject,
    stream: bool,
) -> requests.Response:
    try:
        return requests.post(
            responses_url,
            headers=headers,
            json=request,
            timeout=timeout,
            stream=stream,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Model provider request failed: {exc}") from exc


def _parse_json_object(
    *,
    parse: Callable[[], object],
    invalid_json_message: str,
    non_object_message: str,
) -> JSONObject:
    try:
        payload = parse()
    except ValueError as exc:
        raise RuntimeError(invalid_json_message) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"{non_object_message}: {_json_detail(cast(JSONValue, payload))}"
        )
    return cast(JSONObject, payload)


def _iter_sse_events(response: requests.Response) -> Iterator[tuple[str | None, str]]:
    event_name: str | None = None
    data_lines: list[str] = []

    for raw_line in response.iter_lines():
        line = raw_line.decode("utf-8")
        if not line:
            if data_lines:
                yield event_name, "\n".join(data_lines)
                event_name = None
                data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line.removeprefix("event:").strip() or None
            continue
        if line.startswith("data:"):
            data = line.removeprefix("data:")
            if data.startswith(" "):
                data = data[1:]
            data_lines.append(data)

    if data_lines:
        yield event_name, "\n".join(data_lines)


def _response_detail(response: requests.Response) -> JSONValue:
    try:
        payload = response.json()
    except ValueError:
        return response.text
    return cast(JSONValue, payload)


def _failure_message(*, status_code: int, detail: JSONValue) -> str:
    return f"Model provider request failed: {status_code} {_json_detail(detail)}"


def _json_detail(detail: JSONValue) -> str:
    if isinstance(detail, str):
        return detail
    return json.dumps(detail, separators=(",", ":"))
