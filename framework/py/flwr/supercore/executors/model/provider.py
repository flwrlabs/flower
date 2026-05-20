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
_RESPONSES_PATH = "/responses"
_STREAM_CONTENT_TYPE = "text/event-stream"
_TERMINAL_SUCCESS_EVENTS = frozenset({"response.completed"})
_TERMINAL_FAILURE_EVENTS = frozenset({"response.failed", "response.error", "error"})


@dataclass(frozen=True)
class ModelProviderConfig:
    """Resolved Open Responses provider configuration."""

    base_url: str
    headers: dict[str, str]
    provider_name: str
    timeout_s: float


@dataclass(frozen=True)
class ModelProviderFailurePayload:
    """Structured private payload for model provider failures."""

    provider_name: str
    message: str
    status_code: int | None = None
    detail: JSONValue | None = None
    event: JSONObject | None = None

    def to_dict(self) -> JSONObject:
        """Return this failure payload as JSON-compatible data."""
        payload: JSONObject = {
            "provider_name": self.provider_name,
            "message": self.message,
        }
        if self.status_code is not None:
            payload["status_code"] = self.status_code
        if self.detail is not None:
            payload["detail"] = self.detail
        if self.event is not None:
            payload["event"] = self.event
        return payload


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


def invoke_responses_model(
    request: JSONObject,
    *,
    on_stream_event: Callable[[JSONObject], None] | None = None,
) -> ModelProviderResult:
    """Invoke the configured Open Responses-compatible provider.

    Parameters
    ----------
    request : JSONObject
        Responses-compatible request JSON. Streaming is enabled by setting
        ``stream`` to ``True`` in this request.
    on_stream_event : Callable[[JSONObject], None] | None
        Optional callback invoked once for each parsed provider stream event.

    Returns
    -------
    ModelProviderResult
        Final Responses-compatible JSON object and collected provider events.
    """
    config = resolve_model_provider_config()
    payload = dict(request)
    if payload.get("stream") is True:
        return _invoke_streaming_response(
            config=config,
            request=payload,
            on_stream_event=on_stream_event,
        )
    return _invoke_response(config=config, request=payload)


def resolve_model_provider_config() -> ModelProviderConfig:
    """Resolve Open Responses provider configuration from environment variables."""
    api_key = _model_api_key()
    if api_key is None:
        raise _provider_error(
            status_code=None,
            detail=None,
            message="Model API key is not set (FLWR_MODEL_API_KEY).",
        )
    return ModelProviderConfig(
        base_url=_model_api_endpoint(),
        headers={"Authorization": f"Bearer {api_key}"},
        provider_name=_PROVIDER_NAME,
        timeout_s=_model_timeout_seconds(),
    )


def _invoke_response(
    *,
    config: ModelProviderConfig,
    request: JSONObject,
) -> ModelProviderResult:
    response = _post_responses_request(config=config, request=request, stream=False)
    _raise_for_provider(response=response, provider_name=config.provider_name)
    data = _decode_response_json(response=response, provider_name=config.provider_name)
    _raise_if_provider_payload_failed(
        payload=data,
        status_code=response.status_code,
        provider_name=config.provider_name,
    )
    return ModelProviderResult(response=data, events=[])


def _invoke_streaming_response(
    *,
    config: ModelProviderConfig,
    request: JSONObject,
    on_stream_event: Callable[[JSONObject], None] | None,
) -> ModelProviderResult:
    request_with_stream = dict(request)
    request_with_stream["stream"] = True
    response = _post_responses_request(
        config=config,
        request=request_with_stream,
        stream=True,
    )
    _raise_for_provider(response=response, provider_name=config.provider_name)
    _raise_on_non_sse_stream_response(
        response=response,
        provider_name=config.provider_name,
    )

    events: list[JSONObject] = []
    saw_sse_event = False
    last_event: JSONObject | None = None
    last_data_snippet: str | None = None

    for event_name, data in _iter_sse_events(response):
        saw_sse_event = True
        last_data_snippet = _truncate_snippet(data)
        if data == "[DONE]":
            break

        payload = _parse_stream_payload(data)
        if payload is None:
            continue

        event = _provider_event(event_name=event_name, payload=payload)
        last_event = event
        events.append(event)
        if on_stream_event is not None:
            on_stream_event(event)

        _raise_if_provider_payload_failed(
            payload=event,
            status_code=response.status_code,
            provider_name=config.provider_name,
        )

        event_type = _event_type(event=event, event_name=event_name)
        if event_type in _TERMINAL_FAILURE_EVENTS:
            raise _provider_error(
                status_code=response.status_code,
                detail=_extract_error_detail(event) or event,
                message=_provider_failure_message(
                    provider_name=config.provider_name,
                    status_code=response.status_code,
                    detail=_extract_error_detail(event) or event,
                ),
                event=event,
                provider_name=config.provider_name,
            )
        if event_type in _TERMINAL_SUCCESS_EVENTS:
            return ModelProviderResult(
                response=_final_response_from_event(event),
                events=events,
            )

    raise _provider_error(
        status_code=response.status_code,
        detail=_stream_end_detail(
            response=response,
            saw_sse_event=saw_sse_event,
            last_event=last_event,
            last_data_snippet=last_data_snippet,
        ),
        message="Open Responses stream ended without a terminal event.",
        event=last_event,
        provider_name=config.provider_name,
    )


def _post_responses_request(
    *,
    config: ModelProviderConfig,
    request: JSONObject,
    stream: bool,
) -> requests.Response:
    try:
        return requests.post(
            _endpoint_url(config.base_url),
            headers=config.headers,
            json=request,
            timeout=config.timeout_s,
            stream=stream,
        )
    except requests.RequestException as exc:
        raise _provider_error(
            status_code=None,
            detail=str(exc),
            message=f"{config.provider_name} request failed: {exc}",
            provider_name=config.provider_name,
        ) from exc


def _endpoint_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}{_RESPONSES_PATH}"


def _model_api_key() -> str | None:
    api_key = os.getenv("FLWR_MODEL_API_KEY", "").strip()
    return api_key or None


def _model_api_endpoint() -> str:
    base_url = os.getenv("FLWR_MODEL_API_ENDPOINT", "").strip()
    if not base_url:
        base_url = DEFAULT_MODEL_API_ENDPOINT
    return base_url.rstrip("/")


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


def _raise_for_provider(*, response: requests.Response, provider_name: str) -> None:
    if response.status_code < 400:
        return
    detail = _response_error_detail(response)
    raise _provider_error(
        status_code=response.status_code,
        detail=detail,
        message=_provider_failure_message(
            provider_name=provider_name,
            status_code=response.status_code,
            detail=detail,
        ),
        provider_name=provider_name,
    )


def _decode_response_json(
    *, response: requests.Response, provider_name: str
) -> JSONObject:
    try:
        payload = response.json()
    except ValueError as exc:
        raise _provider_error(
            status_code=response.status_code,
            detail=_response_text_snippet(response),
            message=f"{provider_name} returned invalid JSON.",
            provider_name=provider_name,
        ) from exc
    if not isinstance(payload, dict):
        raise _provider_error(
            status_code=response.status_code,
            detail=cast(JSONValue, payload),
            message=f"{provider_name} returned a non-object JSON payload.",
            provider_name=provider_name,
        )
    return cast(JSONObject, payload)


def _raise_if_provider_payload_failed(
    *, payload: JSONObject, status_code: int, provider_name: str
) -> None:
    detail = _extract_error_detail(payload)
    if detail is None:
        return
    raise _provider_error(
        status_code=status_code,
        detail=detail,
        message=_provider_failure_message(
            provider_name=provider_name,
            status_code=status_code,
            detail=detail,
        ),
        event=payload,
        provider_name=provider_name,
    )


def _iter_sse_events(response: requests.Response) -> Iterator[tuple[str | None, str]]:
    event_name: str | None = None
    data_lines: list[str] = []

    for raw_line in response.iter_lines():
        if raw_line is None:
            continue

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
            data_lines.append(line.removeprefix("data:").lstrip())
            continue

    if data_lines:
        yield event_name, "\n".join(data_lines)


def _parse_stream_payload(data: str) -> JSONObject | None:
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return cast(JSONObject, payload)
    return None


def _provider_event(*, event_name: str | None, payload: JSONObject) -> JSONObject:
    event = dict(payload)
    if isinstance(event.get("type"), str):
        return event
    if event_name is not None:
        event["type"] = event_name
    return event


def _event_type(*, event: JSONObject, event_name: str | None) -> str | None:
    event_type = event.get("type")
    if isinstance(event_type, str):
        return event_type
    return event_name


def _final_response_from_event(event: JSONObject) -> JSONObject:
    response = event.get("response")
    if isinstance(response, dict):
        return cast(JSONObject, response)
    return event


def _raise_on_non_sse_stream_response(
    *, response: requests.Response, provider_name: str
) -> None:
    content_type = _response_content_type(response)
    if content_type and _STREAM_CONTENT_TYPE in content_type:
        return
    if not content_type:
        return

    detail = _non_sse_error_detail(response)
    raise _provider_error(
        status_code=response.status_code,
        detail=detail,
        message=_provider_failure_message(
            provider_name=provider_name,
            status_code=response.status_code,
            detail=detail,
        ),
        provider_name=provider_name,
    )


def _response_content_type(response: requests.Response) -> str:
    headers = getattr(response, "headers", {})
    if not isinstance(headers, Mapping):
        return ""
    return str(headers.get("Content-Type") or "").lower()


def _non_sse_error_detail(response: requests.Response) -> JSONValue:
    try:
        payload = response.json()
    except ValueError:
        body = _response_text_snippet(response)
        return body or "Provider returned non-SSE payload in stream mode."
    if isinstance(payload, dict):
        detail = _extract_error_detail(cast(JSONObject, payload))
        if detail is not None:
            return detail
        return cast(JSONObject, payload)
    return cast(JSONValue, payload)


def _response_error_detail(response: requests.Response) -> JSONValue:
    try:
        payload = response.json()
    except ValueError:
        return _response_text_snippet(response)
    if isinstance(payload, dict):
        detail = _extract_error_detail(cast(JSONObject, payload))
        if detail is not None:
            return detail
        return cast(JSONObject, payload)
    return cast(JSONValue, payload)


def _extract_error_detail(payload: JSONObject) -> JSONValue | None:
    error_obj = payload.get("error")
    if isinstance(error_obj, dict):
        message = error_obj.get("message")
        if isinstance(message, str) and message:
            return message
        return cast(JSONObject, error_obj)
    if isinstance(error_obj, str) and error_obj:
        return error_obj

    detail = payload.get("detail")
    if isinstance(detail, str) and detail:
        return detail

    response_obj = payload.get("response")
    if isinstance(response_obj, dict):
        response_detail = _extract_error_detail(cast(JSONObject, response_obj))
        if response_detail is not None:
            return response_detail

    event_type = payload.get("type")
    if isinstance(event_type, str) and (
        event_type in _TERMINAL_FAILURE_EVENTS or "error" in event_type.lower()
    ):
        for key in ("message", "details", "error_message"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value

    status_value = payload.get("status")
    if isinstance(status_value, str) and status_value.lower() in {"error", "failed"}:
        for key in ("message", "details", "error_message"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        return payload

    return None


def _stream_end_detail(
    *,
    response: requests.Response,
    saw_sse_event: bool,
    last_event: JSONObject | None,
    last_data_snippet: str | None,
) -> JSONValue:
    if last_event is not None:
        return last_event
    if last_data_snippet:
        return last_data_snippet
    if saw_sse_event:
        return "Provider stream ended without a terminal event."
    body = _response_text_snippet(response)
    return body or "Provider stream ended without SSE events."


def _provider_failure_message(
    *, provider_name: str, status_code: int | None, detail: JSONValue
) -> str:
    status = "unknown" if status_code is None else str(status_code)
    return f"{provider_name} request failed: {status} {_json_detail(detail)}"


def _provider_error(
    *,
    status_code: int | None,
    detail: JSONValue | None,
    message: str,
    event: JSONObject | None = None,
    provider_name: str = _PROVIDER_NAME,
) -> ModelProviderError:
    return ModelProviderError(
        ModelProviderFailurePayload(
            provider_name=provider_name,
            status_code=status_code,
            message=message,
            detail=detail,
            event=event,
        )
    )


def _response_text_snippet(response: requests.Response, max_chars: int = 400) -> str:
    try:
        text = str(response.text)
    except Exception:  # pylint: disable=broad-exception-caught
        return "<unavailable>"
    return _truncate_snippet(text, max_chars=max_chars)


def _truncate_snippet(raw: str, max_chars: int = 400) -> str:
    normalized = raw.strip().replace("\n", "\\n")
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[:max_chars]}..."


def _json_detail(detail: JSONValue) -> str:
    if isinstance(detail, str):
        return detail
    return json.dumps(detail, separators=(",", ":"))
