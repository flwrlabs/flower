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
"""Tests for the private Open Responses model provider client."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import Mock

import pytest
import requests

from .provider import (
    JSONObject,
    ModelProviderError,
    invoke_responses_model,
    resolve_model_provider_config,
)


class _Response:
    def __init__(
        self,
        *,
        status_code: int = 200,
        body: object | None = None,
        text: str = "",
        headers: dict[str, str] | None = None,
        lines: list[bytes] | None = None,
        json_error: ValueError | None = None,
    ) -> None:
        self.status_code = status_code
        self._body = body
        self.text = text
        self.headers = headers or {}
        self._lines = lines or []
        self._json_error = json_error

    def json(self) -> object:
        """Return the mocked JSON response body."""
        if self._json_error is not None:
            raise self._json_error
        return self._body

    def iter_lines(self) -> Iterator[bytes]:
        """Return the mocked SSE lines."""
        return iter(self._lines)


def test_resolve_model_provider_config_uses_default_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provider should use the default Flower endpoint."""
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "fk_test")
    monkeypatch.delenv("FLWR_MODEL_API_ENDPOINT", raising=False)
    monkeypatch.delenv("FLWR_MODEL_API_TIMEOUT_S", raising=False)

    config = resolve_model_provider_config()

    assert config.base_url == "https://api.flower.ai/v1"
    assert config.headers == {"Authorization": "Bearer fk_test"}
    assert config.provider_name == "Open Responses"
    assert config.timeout_s == 60.0


def test_resolve_model_provider_config_uses_env_endpoint_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provider should use the configured generic model env vars."""
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "fk_test")
    monkeypatch.setenv("FLWR_MODEL_API_ENDPOINT", "https://example.test/v1/")
    monkeypatch.setenv("FLWR_MODEL_API_TIMEOUT_S", "0.1")

    config = resolve_model_provider_config()

    assert config.base_url == "https://example.test/v1"
    assert config.timeout_s == 1.0


def test_resolve_model_provider_config_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing API key should raise a structured provider error."""
    monkeypatch.delenv("FLWR_MODEL_API_KEY", raising=False)

    with pytest.raises(ModelProviderError) as exc_info:
        resolve_model_provider_config()

    assert exc_info.value.payload.to_dict() == {
        "provider_name": "Open Responses",
        "message": "Model API key is not set (FLWR_MODEL_API_KEY).",
    }


def test_invoke_responses_model_posts_non_streaming_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-streaming calls should post the request unchanged to /responses."""
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "fk_test")
    post_mock = Mock(
        return_value=_Response(
            body={"id": "resp_1", "object": "response", "output_text": "ok"}
        )
    )
    monkeypatch.setattr(
        "flwr.supercore.executors.model.provider.requests.post",
        post_mock,
    )
    request: JSONObject = {
        "model": "flwrlabs/model",
        "input": [{"role": "user", "content": "hi"}],
    }

    result = invoke_responses_model(request)

    assert result.response == {
        "id": "resp_1",
        "object": "response",
        "output_text": "ok",
    }
    assert result.events == []
    assert request == {
        "model": "flwrlabs/model",
        "input": [{"role": "user", "content": "hi"}],
    }
    assert post_mock.call_args.args[0] == "https://api.flower.ai/v1/responses"
    assert post_mock.call_args.kwargs["headers"] == {"Authorization": "Bearer fk_test"}
    assert post_mock.call_args.kwargs["json"] == request
    assert post_mock.call_args.kwargs["stream"] is False


def test_invoke_responses_model_raises_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider HTTP errors should become structured provider errors."""
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "fk_test")
    post_mock = Mock(
        return_value=_Response(
            status_code=400,
            body={"error": {"message": "model not found"}},
            text='{"error":{"message":"model not found"}}',
        )
    )
    monkeypatch.setattr(
        "flwr.supercore.executors.model.provider.requests.post",
        post_mock,
    )

    with pytest.raises(ModelProviderError) as exc_info:
        invoke_responses_model({"model": "missing", "input": []})

    assert exc_info.value.payload.provider_name == "Open Responses"
    assert exc_info.value.payload.status_code == 400
    assert exc_info.value.payload.detail == "model not found"
    assert "Open Responses request failed: 400 model not found" == str(exc_info.value)


def test_invoke_responses_model_raises_on_error_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP 200 payloads containing provider errors should fail."""
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "fk_test")
    post_mock = Mock(
        return_value=_Response(body={"error": {"message": "endpoint paused"}})
    )
    monkeypatch.setattr(
        "flwr.supercore.executors.model.provider.requests.post",
        post_mock,
    )

    with pytest.raises(ModelProviderError) as exc_info:
        invoke_responses_model({"model": "model", "input": []})

    assert exc_info.value.payload.status_code == 200
    assert exc_info.value.payload.detail == "endpoint paused"


def test_invoke_responses_model_collects_stream_events_and_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming calls should collect parsed Open Responses provider events."""
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "fk_test")
    lines = [
        b"event: response.created",
        b'data: {"type":"response.created","response":{"id":"resp_1"}}',
        b"",
        b"event: response.output_text.delta",
        b'data: {"delta":"hel"}',
        b"",
        b"event: response.in_progress",
        b'data: {"sequence_number":2}',
        b"",
        b"event: response.completed",
        b'data: {"type":"response.completed","response":{"id":"resp_1",'
        b'"output_text":"hel"}}',
        b"",
    ]
    post_mock = Mock(
        return_value=_Response(
            headers={"Content-Type": "text/event-stream"},
            lines=lines,
        )
    )
    monkeypatch.setattr(
        "flwr.supercore.executors.model.provider.requests.post",
        post_mock,
    )
    streamed_events: list[JSONObject] = []

    result = invoke_responses_model(
        {"model": "model", "input": [], "stream": True},
        on_stream_event=streamed_events.append,
    )

    assert result.response == {"id": "resp_1", "output_text": "hel"}
    assert result.events == streamed_events
    assert result.events == [
        {"type": "response.created", "response": {"id": "resp_1"}},
        {"delta": "hel", "type": "response.output_text.delta"},
        {"sequence_number": 2, "type": "response.in_progress"},
        {
            "type": "response.completed",
            "response": {"id": "resp_1", "output_text": "hel"},
        },
    ]
    assert post_mock.call_args.kwargs["json"] == {
        "model": "model",
        "input": [],
        "stream": True,
    }
    assert post_mock.call_args.kwargs["stream"] is True


def test_invoke_responses_model_raises_on_stream_error_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider stream error events should become structured provider errors."""
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "fk_test")
    post_mock = Mock(
        return_value=_Response(
            headers={"Content-Type": "text/event-stream"},
            lines=[
                b"event: error",
                b'data: {"message":"model not found"}',
                b"",
            ],
        )
    )
    monkeypatch.setattr(
        "flwr.supercore.executors.model.provider.requests.post",
        post_mock,
    )
    streamed_events: list[JSONObject] = []

    with pytest.raises(ModelProviderError) as exc_info:
        invoke_responses_model(
            {"model": "missing", "input": [], "stream": True},
            on_stream_event=streamed_events.append,
        )

    assert streamed_events == [{"message": "model not found", "type": "error"}]
    assert exc_info.value.payload.status_code == 200
    assert exc_info.value.payload.detail == "model not found"
    assert exc_info.value.payload.event == {
        "message": "model not found",
        "type": "error",
    }


def test_invoke_responses_model_extracts_nested_stream_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Response failure events should expose nested response error messages."""
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "fk_test")
    post_mock = Mock(
        return_value=_Response(
            headers={"Content-Type": "text/event-stream"},
            lines=[
                b"event: response.failed",
                b'data: {"type":"response.failed","response":{"id":"resp_1",'
                b'"status":"failed","error":{"message":"quota exceeded"}}}',
                b"",
            ],
        )
    )
    monkeypatch.setattr(
        "flwr.supercore.executors.model.provider.requests.post",
        post_mock,
    )

    with pytest.raises(ModelProviderError) as exc_info:
        invoke_responses_model({"model": "model", "input": [], "stream": True})

    assert exc_info.value.payload.status_code == 200
    assert exc_info.value.payload.detail == "quota exceeded"


def test_invoke_responses_model_raises_on_non_sse_stream_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stream mode should fail when the provider returns explicit non-SSE JSON."""
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "fk_test")
    post_mock = Mock(
        return_value=_Response(
            headers={"Content-Type": "application/json"},
            body={"detail": "Bad Request: endpoint paused"},
        )
    )
    monkeypatch.setattr(
        "flwr.supercore.executors.model.provider.requests.post",
        post_mock,
    )

    with pytest.raises(ModelProviderError) as exc_info:
        invoke_responses_model({"model": "model", "input": [], "stream": True})

    assert exc_info.value.payload.status_code == 200
    assert exc_info.value.payload.detail == "Bad Request: endpoint paused"


def test_invoke_responses_model_raises_when_stream_has_no_terminal_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streams that end without a terminal event should fail."""
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "fk_test")
    post_mock = Mock(
        return_value=_Response(
            headers={"Content-Type": "text/event-stream"},
            lines=[
                b"event: response.created",
                b'data: {"type":"response.created","response":{"id":"resp_1"}}',
                b"",
            ],
        )
    )
    monkeypatch.setattr(
        "flwr.supercore.executors.model.provider.requests.post",
        post_mock,
    )

    with pytest.raises(ModelProviderError) as exc_info:
        invoke_responses_model({"model": "model", "input": [], "stream": True})

    assert (
        str(exc_info.value) == "Open Responses stream ended without a terminal event."
    )
    assert exc_info.value.payload.event == {
        "type": "response.created",
        "response": {"id": "resp_1"},
    }


def test_invoke_responses_model_raises_on_request_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transport exceptions should become structured provider errors."""
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "fk_test")
    post_mock = Mock(side_effect=requests.Timeout("timed out"))
    monkeypatch.setattr(
        "flwr.supercore.executors.model.provider.requests.post",
        post_mock,
    )

    with pytest.raises(ModelProviderError) as exc_info:
        invoke_responses_model({"model": "model", "input": []})

    assert exc_info.value.payload.status_code is None
    assert exc_info.value.payload.detail == "timed out"
    assert "timed out" in str(exc_info.value)
