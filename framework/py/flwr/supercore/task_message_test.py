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
"""Task-message wrapper tests."""


import json
from typing import cast

import pytest

from flwr.app.metadata import Metadata
from flwr.common import ConfigRecord, Message, RecordDict
from flwr.common.message import make_message
from flwr.supercore.task_message import (
    JsonObject,
    JsonValue,
    ModelTaskMessage,
    ModelTaskResultMessage,
    TaskMessageSpec,
)


def test_task_message_spec_roundtrips_through_message() -> None:
    """TaskMessageSpec should round-trip through Message content."""
    spec = TaskMessageSpec(
        dst_task_id=12,
        message_type="query.model",
        payload={"input": "hello", "model": "gpt-4.1-mini", "stream": True},
        reply_to_message_id="request-id",
        ttl=42.0,
    )

    message = spec.to_message()
    parsed = TaskMessageSpec.from_message(message)

    assert parsed == spec
    assert message.metadata.run_id == 0
    assert message.metadata.message_id == ""
    assert message.metadata.src_node_id == 0
    assert message.metadata.dst_node_id == 0
    assert message.metadata.src_task_id is None
    assert message.metadata.dst_task_id == 12
    assert message.metadata.reply_to_message_id == "request-id"


def test_task_message_spec_serializes_compact_json_payload() -> None:
    """TaskMessageSpec should store payload JSON in a ConfigRecord."""
    spec = TaskMessageSpec(
        dst_task_id=12,
        message_type="query.model",
        payload={"model": "gpt-4.1-mini", "stream": False},
    )

    message = spec.to_message()
    record = message.content.config_records["payload"]

    assert record["json"] == '{"model":"gpt-4.1-mini","stream":false}'


def test_model_task_message_uses_responses_compatible_payload() -> None:
    """ModelTaskMessage should preserve Responses-compatible request fields."""
    spec = ModelTaskMessage.create(
        dst_task_id=7,
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            }
        ],
        model="gpt-4.1-mini",
        stream=True,
        tools=[{"type": "web_search_preview"}],
        tool_choice="auto",
        reasoning={"effort": "low"},
        previous_response_id="resp_prev",
        instructions="Be brief.",
        max_output_tokens=128,
        metadata={"conversation_id": "conv-1"},
        text={"format": {"type": "text"}},
    )

    parsed = ModelTaskMessage.from_message(spec.to_message())

    assert parsed == spec
    assert parsed.message_type == "query.model"
    assert parsed.payload["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello"}],
        }
    ]
    assert parsed.payload["tools"] == [{"type": "web_search_preview"}]
    assert parsed.payload["tool_choice"] == "auto"
    assert parsed.payload["max_output_tokens"] == 128


def test_model_task_result_message_preserves_provider_response() -> None:
    """ModelTaskResultMessage should preserve the provider response JSON."""
    response: JsonObject = {
        "id": "resp_123",
        "object": "response",
        "output": cast(JsonValue, [{"type": "message", "content": []}]),
        "usage": {"input_tokens": 3, "output_tokens": 5},
    }
    spec = ModelTaskResultMessage.create(
        dst_task_id=8,
        response=response,
        response_id="resp_123",
        usage={"input_tokens": 3, "output_tokens": 5},
        finish_reason="stop",
        output=cast(JsonValue, [{"type": "message", "content": []}]),
        events=[{"type": "response.completed"}],
        reply_to_message_id="model-request-message-id",
    )

    parsed = ModelTaskResultMessage.from_message(spec.to_message())

    assert parsed == spec
    assert parsed.message_type == "query.model_result"
    assert parsed.payload["response"] == response
    assert parsed.payload["response_id"] == "resp_123"
    assert parsed.reply_to_message_id == "model-request-message-id"


def test_model_task_message_rejects_wrong_message_type() -> None:
    """ModelTaskMessage should reject messages with the wrong message type."""
    message = ModelTaskResultMessage.create(
        dst_task_id=1,
        response={"id": "resp_123"},
    ).to_message()

    with pytest.raises(ValueError, match="Expected message type query.model"):
        ModelTaskMessage.from_message(message)


@pytest.mark.parametrize(
    ("content", "match"),
    [
        (RecordDict(), "`payload` record"),
        (
            RecordDict({"payload": ConfigRecord({"json": 1})}),
            "`json` field must be a string",
        ),
        (
            RecordDict({"payload": ConfigRecord({"json": "{not-json"})}),
            "valid JSON",
        ),
        (
            RecordDict({"payload": ConfigRecord({"json": "[]"})}),
            "must be an object",
        ),
    ],
)
def test_task_message_spec_rejects_malformed_payload_content(
    content: RecordDict, match: str
) -> None:
    """TaskMessageSpec should reject malformed payload records."""
    message = _make_message("query.model", content=content, dst_task_id=1)

    with pytest.raises(ValueError, match=match):
        ModelTaskMessage.from_message(message)


def test_task_message_spec_rejects_message_without_destination_task() -> None:
    """TaskMessageSpec should require dst_task_id metadata."""
    message = _make_message(
        "query.model",
        content=RecordDict(
            {
                "payload": ConfigRecord(
                    {
                        "json": json.dumps(
                            {"input": "hello", "model": "m", "stream": True}
                        )
                    }
                )
            }
        ),
    )

    with pytest.raises(ValueError, match="dst_task_id"):
        ModelTaskMessage.from_message(message)


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "m", "stream": True},
        {"input": "hello", "stream": True},
        {"input": "hello", "model": "m"},
        {"input": "hello", "model": "m", "stream": "true"},
        {"input": "hello", "model": "m", "stream": True, "max_output_tokens": True},
    ],
)
def test_model_task_message_rejects_invalid_request_payload(
    payload: JsonObject,
) -> None:
    """ModelTaskMessage should validate required request fields."""
    with pytest.raises(ValueError):
        ModelTaskMessage(
            dst_task_id=1,
            message_type="query.model",
            payload=payload,
        )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"response": "resp_123"},
        {"response": {"id": "resp_123"}, "response_id": 123},
        {"response": {"id": "resp_123"}, "usage": []},
        {"response": {"id": "resp_123"}, "events": {}},
    ],
)
def test_model_task_result_message_rejects_invalid_result_payload(
    payload: JsonObject,
) -> None:
    """ModelTaskResultMessage should validate required result fields."""
    with pytest.raises(ValueError):
        ModelTaskResultMessage(
            dst_task_id=1,
            message_type="query.model_result",
            payload=payload,
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": ("not", "json")},
        {1: "non-string-key"},
    ],
)
def test_task_message_spec_rejects_non_json_payload_values(
    payload: dict[object, object],
) -> None:
    """TaskMessageSpec should reject values that are not strict JSON."""
    with pytest.raises(ValueError):
        TaskMessageSpec(
            dst_task_id=1,
            message_type="query.model",
            payload=payload,  # type: ignore[arg-type]
        )


def _make_message(
    message_type: str,
    *,
    content: RecordDict,
    dst_task_id: int | None = None,
) -> Message:
    """Create a Message with task metadata for wrapper tests."""
    metadata = Metadata(
        run_id=0,
        message_id="",
        src_node_id=0,
        dst_node_id=0,
        reply_to_message_id="",
        group_id="",
        created_at=0.0,
        ttl=60.0,
        message_type=message_type,
        dst_task_id=dst_task_id,
    )
    return make_message(metadata=metadata, content=content)
