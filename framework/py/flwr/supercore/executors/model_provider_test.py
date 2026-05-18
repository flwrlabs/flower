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
"""Tests for Responses-compatible model provider runner."""


from __future__ import annotations

from typing import Any

import pytest
import requests

from flwr.supercore.executors import model_provider
from flwr.supercore.executors.model_provider import ModelProviderError
from flwr.supercore.task_message import JsonObject


class FakeResponse:
    """Minimal requests.Response test double."""

    def __init__(
        self,
        json_body: object,
        *,
        status_code: int = 200,
        text: str = "",
        lines: list[str] | None = None,
    ) -> None:
        self._json_body = json_body
        self.status_code = status_code
        self.text = text
        self._lines = lines or []
        self.closed = False

    def json(self) -> object:
        """Return configured JSON body."""
        if isinstance(self._json_body, ValueError):
            raise self._json_body
        return self._json_body

    def iter_lines(self, decode_unicode: bool = False) -> list[str]:
        """Return configured streaming lines."""
        del decode_unicode
        return self._lines

    def close(self) -> None:
        """Mark the response closed."""
        self.closed = True


def test_invoke_responses_model_posts_non_streaming_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test non-streaming provider invocation."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.example/v1")
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return FakeResponse({"id": "resp-1", "output": []})

    monkeypatch.setattr(requests, "post", fake_post)
    request: JsonObject = {"model": "gpt-4.1-mini", "input": [], "stream": False}

    result = model_provider.invoke_responses_model(request)

    assert result.response == {"id": "resp-1", "output": []}
    assert result.events == []
    assert calls[0]["url"] == "https://provider.example/v1/responses"
    assert calls[0]["json"] == request
    assert calls[0]["headers"]["Authorization"] == "Bearer test-key"
    assert "stream" not in calls[0]


def test_invoke_responses_model_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test missing API key is surfaced as a structured provider error."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ModelProviderError) as err:
        model_provider.invoke_responses_model(
            {"model": "gpt-4.1-mini", "input": [], "stream": False}
        )

    assert err.value.error["type"] == "missing_api_key"


def test_invoke_responses_model_handles_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test HTTP errors become structured provider errors."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        del url, kwargs
        return FakeResponse(
            {"error": {"message": "bad request"}},
            status_code=400,
            text="bad request",
        )

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(ModelProviderError) as err:
        model_provider.invoke_responses_model(
            {"model": "gpt-4.1-mini", "input": [], "stream": False}
        )

    assert err.value.error["type"] == "http_error"
    assert err.value.error["status_code"] == 400
    assert err.value.error["response"] == {"error": {"message": "bad request"}}


def test_invoke_responses_model_handles_malformed_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test malformed provider JSON becomes a structured provider error."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        del url, kwargs
        return FakeResponse(ValueError("malformed json"))

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(ModelProviderError) as err:
        model_provider.invoke_responses_model(
            {"model": "gpt-4.1-mini", "input": [], "stream": False}
        )

    assert err.value.error["type"] == "invalid_response"


def test_invoke_responses_model_collects_stream_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test streaming provider invocation collects and forwards SSE events."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_response = FakeResponse(
        {},
        lines=[
            'data: {"type":"response.output_text.delta","delta":"hi"}',
            'data: {"type":"response.completed","response":{"id":"resp-1"}}',
            "data: [DONE]",
        ],
    )
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return fake_response

    monkeypatch.setattr(requests, "post", fake_post)
    seen_events: list[JsonObject] = []

    result = model_provider.invoke_responses_model(
        {"model": "gpt-4.1-mini", "input": [], "stream": True},
        on_stream_event=seen_events.append,
    )

    assert result.response == {"id": "resp-1"}
    assert [event["type"] for event in result.events] == [
        "response.output_text.delta",
        "response.completed",
    ]
    assert seen_events == result.events
    assert calls[0]["stream"] is True
    assert fake_response.closed
