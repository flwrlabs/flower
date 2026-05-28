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
"""Private Responses-compatible provider client for Model executors."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator, Mapping
from typing import cast

import requests

from flwr.supercore.typing import JSONObject, JSONValue

DEFAULT_MODEL_API_ENDPOINT = "https://api.flower.ai/v1/responses"
DEFAULT_MODEL_API_TIMEOUT_S = 180.0
_STREAM_CONTENT_TYPE = "text/event-stream"
_TERMINAL_SUCCESS_EVENTS = frozenset({"response.completed", "response.incomplete"})
_TERMINAL_FAILURE_EVENTS = frozenset({"error", "response.failed"})


def invoke_responses_model(
    request: JSONObject,
    *,
    on_stream_event: Callable[[JSONObject], None] | None = None,
) -> JSONObject:
    """Invoke the configured Responses-compatible model provider."""
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
        "FLWR_MODEL_API_TIMEOUT_S",
        str(DEFAULT_MODEL_API_TIMEOUT_S),
    )
    try:
        timeout_s = float(raw_timeout.strip())
    except ValueError:
        timeout_s = DEFAULT_MODEL_API_TIMEOUT_S
    timeout_s = max(1.0, timeout_s)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = dict(request)
    if payload.get("stream") is True:
        return _invoke_streaming_response(
            responses_url=responses_url,
            headers=headers,
            timeout_s=timeout_s,
            request=payload,
            on_stream_event=on_stream_event,
        )
    return _invoke_response(
        responses_url=responses_url,
        headers=headers,
        timeout_s=timeout_s,
        request=payload,
    )


def _invoke_response(
    *,
    responses_url: str,
    headers: dict[str, str],
    timeout_s: float,
    request: JSONObject,
) -> JSONObject:
    response = _post_responses_request(
        responses_url=responses_url,
        headers=headers,
        timeout_s=timeout_s,
        request=request,
        stream=False,
    )
    _raise_for_http_failure(response)
    response_payload = _decode_response_json(response)
    detail = _extract_error_detail(response_payload)
    if detail is not None:
        raise RuntimeError(
            _failure_message(status_code=response.status_code, detail=detail)
        )
    return response_payload


def _invoke_streaming_response(
    *,
    responses_url: str,
    headers: dict[str, str],
    timeout_s: float,
    request: JSONObject,
    on_stream_event: Callable[[JSONObject], None] | None,
) -> JSONObject:
    request["stream"] = True
    response = _post_responses_request(
        responses_url=responses_url,
        headers=headers,
        timeout_s=timeout_s,
        request=request,
        stream=True,
    )
    _raise_for_http_failure(response)
    _raise_for_non_sse_response(response)

    last_event: JSONObject | None = None
    for event_name, data in _iter_sse_events(response):
        event = _parse_provider_event(event_name=event_name, data=data)
        if event is None:
            continue

        last_event = event
        if on_stream_event is not None:
            on_stream_event(event)

        event_type = event.get("type")
        is_failure_event = (
            isinstance(event_type, str) and event_type in _TERMINAL_FAILURE_EVENTS
        )

        detail = _extract_error_detail(event)
        if detail is not None or is_failure_event:
            raise RuntimeError(
                _failure_message(
                    status_code=response.status_code, detail=detail or event
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
    timeout_s: float,
    request: JSONObject,
    stream: bool,
) -> requests.Response:
    try:
        return requests.post(
            responses_url,
            headers=headers,
            json=request,
            timeout=timeout_s,
            stream=stream,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Model provider request failed: {exc}") from exc


def _raise_for_http_failure(response: requests.Response) -> None:
    if response.status_code < 400:
        return
    detail = _response_detail(response)
    raise RuntimeError(
        _failure_message(status_code=response.status_code, detail=detail)
    )


def _raise_for_non_sse_response(response: requests.Response) -> None:
    headers = getattr(response, "headers", {})
    content_type = (
        str(headers.get("Content-Type") or "").lower()
        if isinstance(headers, Mapping)
        else ""
    )
    if _STREAM_CONTENT_TYPE in content_type:
        return

    detail = _response_detail(response)
    if not content_type and (detail is None or detail == ""):
        detail = "Missing Content-Type header for streaming response."
    raise RuntimeError(
        _failure_message(status_code=response.status_code, detail=detail)
    )


def _decode_response_json(response: requests.Response) -> JSONObject:
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Model provider returned invalid JSON: {_response_text(response)}"
        ) from exc
    if not isinstance(payload, dict):
        detail = cast(JSONValue, payload)
        formatted_detail = (
            detail
            if isinstance(detail, str)
            else json.dumps(detail, separators=(",", ":"))
        )
        raise RuntimeError(
            "Model provider returned a non-object JSON payload: "
            f"{formatted_detail}"
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


def _parse_provider_event(*, event_name: str | None, data: str) -> JSONObject | None:
    if data.strip() == "[DONE]":
        return None

    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Model provider stream returned invalid JSON event: {data}"
        ) from exc
    if not isinstance(payload, dict):
        detail = cast(JSONValue, payload)
        formatted_detail = (
            detail
            if isinstance(detail, str)
            else json.dumps(detail, separators=(",", ":"))
        )
        raise RuntimeError(
            "Model provider stream returned a non-object JSON event: "
            f"{formatted_detail}"
        )

    event = cast(JSONObject, payload)
    if event_name is not None and not isinstance(event.get("type"), str):
        event = dict(event)
        event["type"] = event_name
    return event


def _response_detail(response: requests.Response) -> JSONValue:
    try:
        payload = response.json()
    except ValueError:
        return _response_text(response)
    if isinstance(payload, dict):
        detail = _extract_error_detail(cast(JSONObject, payload))
        return detail if detail is not None else cast(JSONObject, payload)
    return cast(JSONValue, payload)


def _extract_error_detail(payload: JSONObject) -> JSONValue | None:
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        return (
            message if isinstance(message, str) and message else cast(JSONObject, error)
        )
    if isinstance(error, str) and error:
        return error

    detail = payload.get("detail")
    if isinstance(detail, str) and detail:
        return detail

    response = payload.get("response")
    if isinstance(response, dict):
        return _extract_error_detail(cast(JSONObject, response))

    return None


def _failure_message(*, status_code: int, detail: JSONValue) -> str:
    formatted_detail = (
        detail if isinstance(detail, str) else json.dumps(detail, separators=(",", ":"))
    )
    return f"Model provider request failed: {status_code} {formatted_detail}"


def _response_text(response: requests.Response, max_chars: int = 400) -> str:
    try:
        text = str(response.text)
    except Exception:  # pylint: disable=broad-exception-caught
        return "<unavailable>"
    normalized = text.strip().replace("\n", "\\n")
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[:max_chars]}..."
