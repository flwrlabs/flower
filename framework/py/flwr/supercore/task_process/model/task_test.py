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
"""Tests for Model task event buffering."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast
from unittest.mock import Mock

import pytest

from flwr.supercore.json_message.model_message import ModelRequest
from flwr.supercore.typing import JSONObject

from . import task


def _model_request() -> ModelRequest:
    """Create a Model request with routed task metadata."""
    request = ModelRequest(
        dst_task_id=22,
        input_="Return exactly: profiling-ready.",
        model="model",
        stream=True,
    )
    request.metadata.__dict__["_message_id"] = "request-message-id"
    request.metadata.src_task_id = 11
    return request


def _completed_response() -> JSONObject:
    """Return a minimal successful Model response."""
    return {"object": "response", "status": "completed", "output": []}


@pytest.mark.parametrize(
    "first_text_event",
    ["response.output_text.delta", "response.reasoning_summary_text.delta"],
)
def test_handle_task_flushes_first_text_event_eagerly(
    monkeypatch: pytest.MonkeyPatch, first_text_event: str
) -> None:
    """The first text event is persisted without waiting for a full batch."""
    stub = Mock()
    monkeypatch.setattr(
        task, "_pull_model_request", Mock(return_value=_model_request())
    )

    def invoke_provider(
        _request: JSONObject,
        *,
        on_stream_event: Callable[[JSONObject], None],
        usage_recorder: object,
    ) -> JSONObject:
        del usage_recorder
        on_stream_event(cast(JSONObject, {"type": "response.created"}))
        assert stub.PushTaskEvents.call_count == 0
        on_stream_event(
            cast(
                JSONObject,
                {"type": first_text_event, "delta": "Hello"},
            )
        )
        assert stub.PushTaskEvents.call_count == 1
        for index in range(16):
            on_stream_event(cast(JSONObject, {"type": f"response.event-{index}"}))
        return _completed_response()

    monkeypatch.setattr(task, "invoke_model_provider", invoke_provider)

    task.handle_task(client=stub, task_id=22, run_id=7)

    batches = [call.args[0].events for call in stub.PushTaskEvents.call_args_list]
    assert [len(batch) for batch in batches] == [2, 16]
    assert [event.event for event in batches[0]] == [
        "response.created",
        first_text_event,
    ]
