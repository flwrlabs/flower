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
"""Model task message tests."""


import json
from typing import Any

import pytest

from flwr.app.metadata import Metadata
from flwr.common import ConfigRecord, Error, Message, RecordDict
from flwr.common.message import make_message
from flwr.common.serde import message_from_proto, message_to_proto
from flwr.common.serde_utils import metadata_from_proto, metadata_to_proto
from flwr.supercore.date import now
from flwr.supercore.inflatable.inflatable_object import (
    get_object_type_from_object_content,
)
from flwr.supercore.model_message import ModelRequest, ModelResponse


def _metadata(
    *,
    message_type: str,
    src_task_id: int | None = None,
    dst_task_id: int | None = 123,
) -> Metadata:
    """Create metadata for model message tests."""
    return Metadata(
        run_id=0,
        message_id="",
        src_node_id=0,
        dst_node_id=0,
        reply_to_message_id="",
        group_id="",
        created_at=now().timestamp(),
        ttl=3600.0,
        message_type=message_type,
        src_task_id=src_task_id,
        dst_task_id=dst_task_id,
    )


def _message_with_payload(payload: dict[str, Any], *, message_type: str) -> Message:
    """Create a plain Message carrying compact JSON payload."""
    return make_message(
        metadata=_metadata(message_type=message_type),
        content=RecordDict(
            {
                "payload": ConfigRecord(
                    {"json": json.dumps(payload, separators=(",", ":"))}
                )
            }
        ),
    )


def test_metadata_task_ids_roundtrip_through_proto() -> None:
    """Metadata should preserve task IDs through protobuf serde."""
    metadata = _metadata(
        message_type=ModelRequest.MESSAGE_TYPE,
        src_task_id=123,
        dst_task_id=456,
    )

    proto = metadata_to_proto(metadata)
    assert proto.HasField("src_task_id")
    assert proto.HasField("dst_task_id")
    assert proto.src_task_id == 123
    assert proto.dst_task_id == 456

    actual = metadata_from_proto(proto)
    assert actual.src_task_id == 123
    assert actual.dst_task_id == 456


def test_metadata_task_ids_remain_unset_when_absent() -> None:
    """Metadata should not set optional task ID proto fields when absent."""
    proto = metadata_to_proto(
        _metadata(message_type=ModelRequest.MESSAGE_TYPE, dst_task_id=None)
    )

    assert not proto.HasField("src_task_id")
    assert not proto.HasField("dst_task_id")

    actual = metadata_from_proto(proto)
    assert actual.src_task_id is None
    assert actual.dst_task_id is None


def test_model_request_creates_responses_request_payload() -> None:
    """ModelRequest should carry an OpenAI Responses create-request payload."""
    request = ModelRequest(
        dst_task_id=123,
        input=[{"role": "user", "content": "Hello"}],
        model="gpt-5",
        stream=True,
        tools=[{"type": "web_search_preview"}],
        tool_choice="auto",
        reasoning={"effort": "medium"},
        previous_response_id="resp_previous",
        instructions="Be concise.",
        max_output_tokens=100,
        metadata={"conversation_id": "conv_123"},
        text={"format": {"type": "text"}},
        ttl=10.0,
    )

    assert isinstance(request, Message)
    assert request.metadata.message_type == "query.model_request"
    assert request.metadata.run_id == 0
    assert request.metadata.src_task_id is None
    assert request.metadata.dst_task_id == 123
    assert request.metadata.ttl == 10.0
    assert request.payload == {
        "model": "gpt-5",
        "input": [{"role": "user", "content": "Hello"}],
        "stream": True,
        "tools": [{"type": "web_search_preview"}],
        "tool_choice": "auto",
        "reasoning": {"effort": "medium"},
        "previous_response_id": "resp_previous",
        "instructions": "Be concise.",
        "max_output_tokens": 100,
        "metadata": {"conversation_id": "conv_123"},
        "text": {"format": {"type": "text"}},
    }


def test_model_response_creates_responses_object_payload() -> None:
    """ModelResponse should carry the OpenAI Responses object directly."""
    response_payload = {
        "id": "resp_123",
        "object": "response",
        "status": "completed",
        "model": "gpt-5",
        "output": [{"type": "message", "role": "assistant", "content": []}],
        "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    }

    response = ModelResponse(
        dst_task_id=456,
        response=response_payload,
        reply_to_message_id="request-message-id",
    )

    assert isinstance(response, Message)
    assert response.metadata.message_type == "query.model_response"
    assert response.metadata.dst_task_id == 456
    assert response.metadata.reply_to_message_id == "request-message-id"
    assert response.payload == response_payload
    assert "response" not in response.payload


def test_model_messages_roundtrip_through_plain_message_proto() -> None:
    """Typed model messages should roundtrip through message.proto.Message."""
    request = ModelRequest(
        dst_task_id=123,
        input=[{"role": "user", "content": "Hello"}],
        model="gpt-5",
        stream=False,
    )

    plain = message_from_proto(message_to_proto(request))

    assert type(plain) is Message
    parsed = ModelRequest.from_message(plain)
    assert isinstance(parsed, ModelRequest)
    assert parsed.payload == request.payload
    assert parsed.metadata.dst_task_id == 123


def test_model_message_deflates_as_plain_message_transport() -> None:
    """Typed messages should keep the inflatable transport type as Message."""
    request = ModelRequest(
        dst_task_id=123,
        input=[{"role": "user", "content": "Hello"}],
        model="gpt-5",
        stream=True,
    )

    request_bytes = request.deflate()

    assert get_object_type_from_object_content(request_bytes) == Message.__qualname__
    inflated = ModelRequest.inflate(request_bytes, children=request.children)
    assert isinstance(inflated, ModelRequest)
    assert inflated.payload == request.payload


def test_model_request_from_message_rejects_wrong_message_type() -> None:
    """ModelRequest parsing should reject non-request messages."""
    message = _message_with_payload(
        {"model": "gpt-5", "input": [], "stream": True},
        message_type=ModelResponse.MESSAGE_TYPE,
    )

    with pytest.raises(ValueError, match="Expected message type"):
        ModelRequest.from_message(message)


@pytest.mark.parametrize("dst_task_id", ["123", True])
def test_model_request_rejects_invalid_constructor_dst_task_id(
    dst_task_id: Any,
) -> None:
    """ModelRequest should reject invalid destination task IDs immediately."""
    with pytest.raises(ValueError, match="dst_task_id"):
        ModelRequest(
            dst_task_id=dst_task_id,
            input=[],
            model="gpt-5",
            stream=True,
        )


@pytest.mark.parametrize("dst_task_id", ["123", True])
def test_model_response_rejects_invalid_constructor_dst_task_id(
    dst_task_id: Any,
) -> None:
    """ModelResponse should reject invalid destination task IDs immediately."""
    with pytest.raises(ValueError, match="dst_task_id"):
        ModelResponse(dst_task_id=dst_task_id, response={"object": "response"})


@pytest.mark.parametrize(
    "message",
    [
        make_message(
            metadata=_metadata(message_type=ModelRequest.MESSAGE_TYPE),
            content=RecordDict(),
        ),
        make_message(
            metadata=_metadata(message_type=ModelRequest.MESSAGE_TYPE),
            content=RecordDict({"payload": ConfigRecord({"json": 1})}),
        ),
        make_message(
            metadata=_metadata(message_type=ModelRequest.MESSAGE_TYPE),
            content=RecordDict({"payload": ConfigRecord({"json": "{"})}),
        ),
        make_message(
            metadata=_metadata(message_type=ModelRequest.MESSAGE_TYPE),
            content=RecordDict({"payload": ConfigRecord({"json": "[]"})}),
        ),
        make_message(
            metadata=_metadata(message_type=ModelRequest.MESSAGE_TYPE),
            error=Error(code=1, reason="failed"),
        ),
    ],
)
def test_model_request_from_message_rejects_invalid_payload(message: Message) -> None:
    """ModelRequest parsing should reject invalid payload transport shape."""
    with pytest.raises(ValueError):
        ModelRequest.from_message(message)


@pytest.mark.parametrize(
    "payload",
    [
        {"input": [], "stream": True},
        {"model": "gpt-5", "stream": True},
        {"model": "gpt-5", "input": []},
        {"model": "gpt-5", "input": [], "stream": "true"},
        {"model": "gpt-5", "input": [1], "stream": True},
        {"model": "gpt-5", "input": [], "stream": True, "tools": ["x"]},
        {"model": "gpt-5", "input": [], "stream": True, "reasoning": []},
    ],
)
def test_model_request_from_message_rejects_invalid_request_shape(
    payload: dict[str, Any],
) -> None:
    """ModelRequest parsing should validate Responses request fields."""
    with pytest.raises(ValueError):
        ModelRequest.from_message(
            _message_with_payload(payload, message_type=ModelRequest.MESSAGE_TYPE)
        )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"object": "chat.completion"},
        {"object": "response", "id": 123},
        {"object": "response", "output": {}},
        {"object": "response", "output": [1]},
        {"object": "response", "error": "failed"},
    ],
)
def test_model_response_from_message_rejects_invalid_response_shape(
    payload: dict[str, Any],
) -> None:
    """ModelResponse parsing should validate Responses object fields."""
    with pytest.raises(ValueError):
        ModelResponse.from_message(
            _message_with_payload(payload, message_type=ModelResponse.MESSAGE_TYPE)
        )


def test_model_request_rejects_non_json_payload_values() -> None:
    """ModelRequest should reject values that cannot be encoded as JSON."""
    with pytest.raises(ValueError, match="JSON serializable"):
        ModelRequest(
            dst_task_id=123,
            input=[{"role": "user", "content": object()}],
            model="gpt-5",
            stream=True,
        )
