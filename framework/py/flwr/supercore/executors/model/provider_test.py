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
"""Tests for the private model provider client."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from unittest.mock import Mock

import pytest

from flwr.supercore.typing import JSONObject

from .provider import invoke_responses_model


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


def test_invoke_responses_model_rejects_base_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured provider endpoints must include the responses path."""
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "fk_test")
    monkeypatch.setenv("FLWR_MODEL_API_ENDPOINT", "https://example.test/v1")

    with pytest.raises(RuntimeError, match="must include the /responses path"):
        invoke_responses_model({"model": "model", "input": []})


def test_invoke_responses_model_collects_stream_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming calls should collect events and accept incomplete terminals."""
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

    assert result == {"id": "resp_1", "output_text": "hel"}
    assert streamed_events == [
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

    _patch_post(
        monkeypatch,
        _Response(
            headers={"Content-Type": "text/event-stream"},
            lines=[
                b"data: [DONE]",
                b"",
                b"event: response.incomplete",
                b'data: {"type":"response.incomplete","response":{"id":"resp_1",'
                b'"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}',
                b"",
            ],
        ),
    )

    result = invoke_responses_model({"model": "model", "input": [], "stream": True})

    assert result == {
        "id": "resp_1",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
    }


def test_invoke_responses_model_raises_on_stream_failure_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider stream failure events should become structured provider errors."""
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "fk_test")
    cases = [
        (
            [
                b"event: response.failed",
                b'data: {"type":"response.failed","response":{"id":"resp_1",'
                b'"error":{"message":"quota exceeded"}}}',
                b"",
            ],
            "quota exceeded",
        ),
        (
            [
                b"event: error",
                b'data: {"type":"error","error":{"message":"bad request"}}',
                b"",
            ],
            "bad request",
        ),
    ]

    for lines, expected_detail in cases:
        _patch_post(
            monkeypatch,
            _Response(headers={"Content-Type": "text/event-stream"}, lines=lines),
        )

        with pytest.raises(RuntimeError) as exc_info:
            invoke_responses_model({"model": "model", "input": [], "stream": True})

        assert str(exc_info.value) == (
            f"Model provider request failed: 200 {expected_detail}"
        )
