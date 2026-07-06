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

from .provider import (
    DEFAULT_MODEL_API_ENDPOINT,
    ModelProviderError,
    invoke_model_provider,
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
        "flwr.supercore.task_process.model.provider.requests.post",
        post_mock,
    )
    return post_mock


@pytest.mark.parametrize(
    "endpoint",
    [
        None,
        DEFAULT_MODEL_API_ENDPOINT,
        f"{DEFAULT_MODEL_API_ENDPOINT}/",
    ],
)
def test_invoke_model_provider_requires_key_for_default_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str | None,
) -> None:
    """Default model endpoint calls should fail before network without an API key."""
    if endpoint is not None:
        monkeypatch.setenv("FLWR_MODEL_API_ENDPOINT", endpoint)
    post_mock = _patch_post(monkeypatch, _Response(body={"id": "resp_1"}))

    with pytest.raises(RuntimeError, match="FLWR_MODEL_API_KEY"):
        invoke_model_provider({"model": "model", "input": []})

    post_mock.assert_not_called()


def test_invoke_model_provider_omits_auth_for_endpoint_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit endpoints should support proxy calls without bearer auth."""
    monkeypatch.setenv("FLWR_MODEL_API_ENDPOINT", "http://proxy/v1/responses")
    post_mock = _patch_post(monkeypatch, _Response(body={"id": "resp_1"}))

    result = invoke_model_provider({"model": "model", "input": []})

    assert result == {"id": "resp_1"}
    assert post_mock.call_args.args == ("http://proxy/v1/responses",)
    assert post_mock.call_args.kwargs["headers"] == {"Content-Type": "application/json"}


def test_invoke_model_provider_keeps_auth_when_key_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct provider calls should keep sending bearer auth."""
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "fk_test")
    post_mock = _patch_post(monkeypatch, _Response(body={"id": "resp_1"}))

    result = invoke_model_provider({"model": "model", "input": []})

    assert result == {"id": "resp_1"}
    assert post_mock.call_args.kwargs["headers"] == {
        "Authorization": "Bearer fk_test",
        "Content-Type": "application/json",
    }


def test_invoke_model_provider_collects_stream_events(
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

    result = invoke_model_provider(
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

    result = invoke_model_provider({"model": "model", "input": [], "stream": True})

    assert result == {
        "id": "resp_1",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
    }


@pytest.mark.parametrize(
    ("lines", "expected_detail", "expected_message"),
    [
        (
            [
                b"event: response.failed",
                b'data: {"type":"response.failed","response":{"id":"resp_1",'
                b'"error":{"message":"quota exceeded"}}}',
                b"",
            ],
            {
                "type": "response.failed",
                "response": {"id": "resp_1", "error": {"message": "quota exceeded"}},
            },
            (
                'Model provider request failed: 200 {"type":"response.failed",'
                '"response":{"id":"resp_1","error":{"message":"quota exceeded"}}}'
            ),
        ),
        (
            [
                b"event: error",
                b'data: {"type":"error","error":{"message":"bad request"}}',
                b"",
            ],
            {"type": "error", "error": {"message": "bad request"}},
            (
                'Model provider request failed: 200 {"type":"error","error":'
                '{"message":"bad request"}}'
            ),
        ),
    ],
)
def test_invoke_model_provider_raises_on_stream_failure_events(
    monkeypatch: pytest.MonkeyPatch,
    lines: list[bytes],
    expected_detail: JSONObject,
    expected_message: str,
) -> None:
    """Provider stream failure events should become structured provider errors."""
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "fk_test")
    _patch_post(
        monkeypatch,
        _Response(headers={"Content-Type": "text/event-stream"}, lines=lines),
    )

    with pytest.raises(ModelProviderError) as exc_info:
        invoke_model_provider({"model": "model", "input": [], "stream": True})

    assert exc_info.value.status_code == 200
    assert exc_info.value.detail == expected_detail
    assert str(exc_info.value) == expected_message
