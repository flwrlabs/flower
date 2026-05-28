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
"""Private Open Responses provider client for Model executors."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import TypeAlias, cast

import requests

JSONValue: TypeAlias = (
    None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
)
JSONObject: TypeAlias = dict[str, JSONValue]

DEFAULT_MODEL_API_ENDPOINT = "https://api.flower.ai/v1"
DEFAULT_MODEL_API_TIMEOUT_S = 60.0
_PROVIDER_NAME = "Open Responses"
_STREAM_CONTENT_TYPE = "text/event-stream"
_TERMINAL_SUCCESS_EVENTS = frozenset({"response.completed", "response.incomplete"})
_TERMINAL_FAILURE_EVENTS = frozenset({"error", "response.failed"})


@dataclass(frozen=True)
class ModelProviderFailurePayload:
    """Structured private payload for model provider failures."""

    message: str
    status_code: int | None = None
    detail: JSONValue | None = None
    event: JSONObject | None = None


class ModelProviderError(RuntimeError):
    """Raised when the configured Open Responses provider request fails."""

    def __init__(self, payload: ModelProviderFailurePayload) -> None:
        super().__init__(payload.message)
        self.payload = payload


@dataclass(frozen=True)
class ModelProviderResult:
    """Result returned by the Open Responses provider client."""

    response: JSONObject
    events: list[JSONObject]


@dataclass(frozen=True)
class _ProviderConfig:
    responses_url: str
    headers: dict[str, str]
    timeout_s: float


def invoke_responses_model(
    request: JSONObject,
    *,
    on_stream_event: Callable[[JSONObject], None] | None = None,
) -> ModelProviderResult:
    """Invoke the configured Open Responses-compatible provider."""
    config = _resolve_provider_config()
    payload = dict(request)
    if payload.get("stream") is True:
        return _invoke_streaming_response(
            config=config,
            request=payload,
            on_stream_event=on_stream_event,
        )
    return _invoke_response(config=config, request=payload)


def _resolve_provider_config() -> _ProviderConfig:
    api_key = os.getenv("FLWR_MODEL_API_KEY", "").strip()
    if not api_key:
        raise _provider_error("Model API key is not set (FLWR_MODEL_API_KEY).")

    base_url = os.getenv("FLWR_MODEL_API_ENDPOINT", "").strip()
    if not base_url:
        base_url = DEFAULT_MODEL_API_ENDPOINT

    return _ProviderConfig(
        responses_url=_responses_url(base_url),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout_s=_model_timeout_seconds(),
    )


def _responses_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/responses"):
        return endpoint
    return f"{endpoint}/responses"


def _model_timeout_seconds() -> float:
    raw_timeout = os.getenv(
        "FLWR_MODEL_API_TIMEOUT_S",
        str(DEFAULT_MODEL_API_TIMEOUT_S),
    )
    try:
        timeout_s = float(raw_timeout.strip())
    except ValueError:
        timeout_s = DEFAULT_MODEL_API_TIMEOUT_S
    return max(1.0, timeout_s)


def _invoke_response(
    *,
    config: _ProviderConfig,
    request: JSONObject,
) -> ModelProviderResult:
    response = _post_responses_request(config=config, request=request, stream=False)
    _raise_for_http_failure(response)
    payload = _decode_response_json(response)
    _raise_for_payload_failure(payload, status_code=response.status_code)
    return ModelProviderResult(response=payload, events=[])


def _invoke_streaming_response(
    *,
    config: _ProviderConfig,
    request: JSONObject,
    on_stream_event: Callable[[JSONObject], None] | None,
) -> ModelProviderResult:
    request["stream"] = True
    response = _post_responses_request(config=config, request=request, stream=True)
    _raise_for_http_failure(response)
    _raise_for_non_sse_response(response)

    events: list[JSONObject] = []
    last_event: JSONObject | None = None
    for event_name, data in _iter_sse_events(response):
        event = _parse_provider_event(event_name=event_name, data=data)
        if event is None:
            continue

        last_event = event
        events.append(event)
        if on_stream_event is not None:
            on_stream_event(event)

        event_type = event.get("type")
        is_failure_event = (
            isinstance(event_type, str) and event_type in _TERMINAL_FAILURE_EVENTS
        )

        detail = _extract_error_detail(event)
        if detail is not None or is_failure_event:
            raise _provider_error(
                _failure_message(
                    status_code=response.status_code, detail=detail or event
                ),
                status_code=response.status_code,
                detail=detail or event,
                event=event,
            )

        if isinstance(event_type, str) and event_type in _TERMINAL_SUCCESS_EVENTS:
            return ModelProviderResult(
                response=_final_response_from_event(event),
                events=events,
            )

    raise _provider_error(
        "Open Responses stream ended without a terminal event.",
        status_code=response.status_code,
        detail=last_event,
        event=last_event,
    )


def _post_responses_request(
    *,
    config: _ProviderConfig,
    request: JSONObject,
    stream: bool,
) -> requests.Response:
    try:
        return requests.post(
            config.responses_url,
            headers=config.headers,
            json=request,
            timeout=config.timeout_s,
            stream=stream,
        )
    except requests.RequestException as exc:
        raise _provider_error(
            f"{_PROVIDER_NAME} request failed: {exc}",
            detail=str(exc),
        ) from exc


def _raise_for_http_failure(response: requests.Response) -> None:
    if response.status_code < 400:
        return
    detail = _response_detail(response)
    raise _provider_error(
        _failure_message(status_code=response.status_code, detail=detail),
        status_code=response.status_code,
        detail=detail,
    )


def _raise_for_payload_failure(payload: JSONObject, *, status_code: int) -> None:
    detail = _extract_error_detail(payload)
    if detail is None:
        return
    raise _provider_error(
        _failure_message(status_code=status_code, detail=detail),
        status_code=status_code,
        detail=detail,
        event=payload,
    )


def _raise_for_non_sse_response(response: requests.Response) -> None:
    content_type = _response_content_type(response)
    if _STREAM_CONTENT_TYPE in content_type:
        return

    detail = _response_detail(response)
    if not content_type and (detail is None or detail == ""):
        detail = "Missing Content-Type header for streaming response."
    raise _provider_error(
        _failure_message(status_code=response.status_code, detail=detail),
        status_code=response.status_code,
        detail=detail,
    )


def _decode_response_json(response: requests.Response) -> JSONObject:
    try:
        payload = response.json()
    except ValueError as exc:
        raise _provider_error(
            f"{_PROVIDER_NAME} returned invalid JSON.",
            status_code=response.status_code,
            detail=_response_text(response),
        ) from exc
    if not isinstance(payload, dict):
        raise _provider_error(
            f"{_PROVIDER_NAME} returned a non-object JSON payload.",
            status_code=response.status_code,
            detail=cast(JSONValue, payload),
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
        raise _provider_error(
            f"{_PROVIDER_NAME} stream returned invalid JSON event.",
            detail=data,
        ) from exc
    if not isinstance(payload, dict):
        raise _provider_error(
            f"{_PROVIDER_NAME} stream returned a non-object JSON event.",
            detail=cast(JSONValue, payload),
        )

    event = cast(JSONObject, payload)
    if event_name is not None and not isinstance(event.get("type"), str):
        event = dict(event)
        event["type"] = event_name
    return event


def _final_response_from_event(event: JSONObject) -> JSONObject:
    response = event.get("response")
    if isinstance(response, dict):
        return cast(JSONObject, response)
    return event


def _response_content_type(response: requests.Response) -> str:
    headers = getattr(response, "headers", {})
    if not isinstance(headers, Mapping):
        return ""
    return str(headers.get("Content-Type") or "").lower()


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
    return f"{_PROVIDER_NAME} request failed: {status_code} {_format_detail(detail)}"


def _provider_error(
    message: str,
    *,
    status_code: int | None = None,
    detail: JSONValue | None = None,
    event: JSONObject | None = None,
) -> ModelProviderError:
    return ModelProviderError(
        ModelProviderFailurePayload(
            message=message,
            status_code=status_code,
            detail=detail,
            event=event,
        )
    )


def _response_text(response: requests.Response, max_chars: int = 400) -> str:
    try:
        text = str(response.text)
    except Exception:  # pylint: disable=broad-exception-caught
        return "<unavailable>"
    normalized = text.strip().replace("\n", "\\n")
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[:max_chars]}..."


def _format_detail(detail: JSONValue) -> str:
    if isinstance(detail, str):
        return detail
    return json.dumps(detail, separators=(",", ":"))
