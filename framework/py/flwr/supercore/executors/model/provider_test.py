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
from dataclasses import dataclass, field
from unittest.mock import Mock

import pytest
import requests

from .provider import (
    JSONObject,
    ModelProviderError,
    _iter_sse_events,
    invoke_responses_model,
)


@dataclass
class _Response:
    status_code: int = 200
    body: object | None = None
    text: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    lines: list[bytes] = field(default_factory=list)

    def json(self) -> object:
        """Return the mocked JSON response body."""
        return self.body

    def iter_lines(self) -> Iterator[bytes]:
        """Return the mocked stream response lines."""
        return iter(self.lines)


def _patch_post(monkeypatch: pytest.MonkeyPatch, response: _Response) -> Mock:
    post_mock = Mock(return_value=response)
    monkeypatch.setattr(
        "flwr.supercore.executors.model.provider.requests.post",
        post_mock,
    )
    return post_mock


def test_invoke_responses_model_uses_env_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider calls should use the configured endpoint, key, and timeout."""
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "fk_test")
    monkeypatch.setenv("FLWR_MODEL_API_ENDPOINT", "https://example.test/v1/")
    monkeypatch.setenv("FLWR_MODEL_API_TIMEOUT_S", "0.1")
    post_mock = _patch_post(
        monkeypatch,
        _Response(body={"id": "resp_1", "object": "response"}),
    )
    request: JSONObject = {"model": "model", "input": []}

    result = invoke_responses_model(request)

    assert result.response == {"id": "resp_1", "object": "response"}
    assert not result.events
    assert post_mock.call_args.args[0] == "https://example.test/v1/responses"
    assert post_mock.call_args.kwargs["headers"] == {
        "Authorization": "Bearer fk_test",
        "Content-Type": "application/json",
    }
    assert post_mock.call_args.kwargs["json"] == request
    assert post_mock.call_args.kwargs["timeout"] == 1.0
    assert post_mock.call_args.kwargs["stream"] is False


def test_invoke_responses_model_uses_default_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing endpoint should default to the Flower model API."""
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "fk_test")
    monkeypatch.delenv("FLWR_MODEL_API_ENDPOINT", raising=False)
    post_mock = _patch_post(monkeypatch, _Response(body={}))

    invoke_responses_model({"model": "model", "input": []})

    assert post_mock.call_args.args[0] == "https://api.flower.ai/v1/responses"


def test_invoke_responses_model_accepts_full_responses_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured endpoint may be the base URL or the full responses URL."""
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "fk_test")
    monkeypatch.setenv("FLWR_MODEL_API_ENDPOINT", "https://example.test/v1/responses/")
    post_mock = _patch_post(monkeypatch, _Response(body={}))

    invoke_responses_model({"model": "model", "input": []})

    assert post_mock.call_args.args[0] == "https://example.test/v1/responses"


def test_invoke_responses_model_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing API key should raise a structured provider error."""
    monkeypatch.delenv("FLWR_MODEL_API_KEY", raising=False)

    with pytest.raises(ModelProviderError) as exc_info:
        invoke_responses_model({"model": "model", "input": []})

    assert exc_info.value.payload.message == (
        "Model API key is not set (FLWR_MODEL_API_KEY)."
    )


def test_invoke_responses_model_raises_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider HTTP errors should become structured provider errors."""
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "fk_test")
    _patch_post(
        monkeypatch,
        _Response(
            status_code=400,
            body={"error": {"message": "model not found"}},
        ),
    )

    with pytest.raises(ModelProviderError) as exc_info:
        invoke_responses_model({"model": "missing", "input": []})

    assert exc_info.value.payload.status_code == 400
    assert exc_info.value.payload.detail == "model not found"
    assert str(exc_info.value) == "Open Responses request failed: 400 model not found"


def test_invoke_responses_model_raises_on_error_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP 200 payloads containing provider errors should fail."""
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "fk_test")
    _patch_post(
        monkeypatch,
        _Response(body={"error": {"message": "endpoint paused"}}),
    )

    with pytest.raises(ModelProviderError) as exc_info:
        invoke_responses_model({"model": "model", "input": []})

    assert exc_info.value.payload.status_code == 200
    assert exc_info.value.payload.detail == "endpoint paused"


def test_invoke_responses_model_collects_stream_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming calls should collect events and call the stream callback."""
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "fk_test")
    post_mock = _patch_post(
        monkeypatch,
        _Response(
            headers={"Content-Type": "text/event-stream"},
            lines=[
                b"event: response.created",
                b'data: {"type":"response.created","response":{"id":"resp_1"}}',
                b"",
                b"event: response.output_text.delta",
                b'data: {"delta":"hel"}',
                b"",
                b"event: response.completed",
                b'data: {"type":"response.completed","response":{"id":"resp_1",'
                b'"output_text":"hel"}}',
                b"",
            ],
        ),
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


def test_invoke_responses_model_ignores_done_stream_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming calls should ignore the terminal [DONE] marker."""
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "fk_test")
    _patch_post(
        monkeypatch,
        _Response(
            headers={"Content-Type": "text/event-stream"},
            lines=[
                b"data: [DONE]",
                b"",
                b"event: response.completed",
                b'data: {"type":"response.completed","response":{"id":"resp_1"}}',
                b"",
            ],
        ),
    )

    result = invoke_responses_model({"model": "model", "input": [], "stream": True})

    assert result.response == {"id": "resp_1"}
    assert result.events == [
        {"type": "response.completed", "response": {"id": "resp_1"}}
    ]


def test_invoke_responses_model_returns_incomplete_stream_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Incomplete response events should be treated as terminal responses."""
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "fk_test")
    _patch_post(
        monkeypatch,
        _Response(
            headers={"Content-Type": "text/event-stream"},
            lines=[
                b"event: response.incomplete",
                b'data: {"type":"response.incomplete","response":{"id":"resp_1",'
                b'"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}',
                b"",
            ],
        ),
    )

    result = invoke_responses_model({"model": "model", "input": [], "stream": True})

    assert result.response == {
        "id": "resp_1",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
    }


def test_invoke_responses_model_raises_on_stream_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider stream failure events should become structured provider errors."""
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "fk_test")
    _patch_post(
        monkeypatch,
        _Response(
            headers={"Content-Type": "text/event-stream"},
            lines=[
                b"event: response.failed",
                b'data: {"type":"response.failed","response":{"id":"resp_1",'
                b'"error":{"message":"quota exceeded"}}}',
                b"",
            ],
        ),
    )

    with pytest.raises(ModelProviderError) as exc_info:
        invoke_responses_model({"model": "model", "input": [], "stream": True})

    assert exc_info.value.payload.status_code == 200
    assert exc_info.value.payload.detail == "quota exceeded"


def test_invoke_responses_model_raises_on_stream_error_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider stream error events should become structured provider errors."""
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "fk_test")
    _patch_post(
        monkeypatch,
        _Response(
            headers={"Content-Type": "text/event-stream"},
            lines=[
                b"event: error",
                b'data: {"type":"error","error":{"message":"bad request"}}',
                b"",
            ],
        ),
    )

    with pytest.raises(ModelProviderError) as exc_info:
        invoke_responses_model({"model": "model", "input": [], "stream": True})

    assert exc_info.value.payload.status_code == 200
    assert exc_info.value.payload.detail == "bad request"


def test_invoke_responses_model_raises_on_non_sse_stream_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stream mode should fail when the provider returns explicit non-SSE JSON."""
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "fk_test")
    _patch_post(
        monkeypatch,
        _Response(
            headers={"Content-Type": "application/json"},
            body={"detail": "Bad Request: endpoint paused"},
        ),
    )

    with pytest.raises(ModelProviderError) as exc_info:
        invoke_responses_model({"model": "model", "input": [], "stream": True})

    assert exc_info.value.payload.status_code == 200
    assert exc_info.value.payload.detail == "Bad Request: endpoint paused"


def test_invoke_responses_model_raises_on_missing_stream_content_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stream mode should fail when the provider omits the SSE content type."""
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "fk_test")
    _patch_post(monkeypatch, _Response())

    with pytest.raises(ModelProviderError) as exc_info:
        invoke_responses_model({"model": "model", "input": [], "stream": True})

    assert exc_info.value.payload.status_code == 200
    assert exc_info.value.payload.detail == (
        "Missing Content-Type header for streaming response."
    )


def test_invoke_responses_model_raises_on_invalid_stream_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed JSON SSE events should become structured provider errors."""
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "fk_test")
    _patch_post(
        monkeypatch,
        _Response(
            headers={"Content-Type": "text/event-stream"},
            lines=[
                b"data: {bad-json}",
                b"",
            ],
        ),
    )

    with pytest.raises(ModelProviderError) as exc_info:
        invoke_responses_model({"model": "model", "input": [], "stream": True})

    assert exc_info.value.payload.detail == "{bad-json}"
    assert str(exc_info.value) == "Open Responses stream returned invalid JSON event."


def test_iter_sse_events_preserves_data_leading_space() -> None:
    """SSE data parsing should remove at most one optional leading space."""
    response = _Response(
        lines=[
            b"event: response.output_text.delta",
            b"data:  leading-space",
            b"",
        ],
    )

    assert list(_iter_sse_events(response)) == [
        ("response.output_text.delta", " leading-space")
    ]


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
