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
from flwr.common import ConfigRecord, Message, RecordDict
from flwr.common.message import make_message
from flwr.supercore.date import now
from flwr.supercore.model_message import ModelRequest, ModelResponse
from flwr.supercore.typing import JSONObject


def _metadata(
    *,
    message_type: str,
    dst_task_id: int | None = 123,
    reply_to_message_id: str = "",
) -> Metadata:
    """Create metadata for model message tests."""
    return Metadata(
        run_id=0,
        message_id="",
        src_node_id=0,
        dst_node_id=0,
        reply_to_message_id=reply_to_message_id,
        group_id="",
        created_at=now().timestamp(),
        ttl=3600.0,
        message_type=message_type,
        dst_task_id=dst_task_id,
    )


def _message_with_payload(
    payload: dict[str, Any],
    *,
    message_type: str,
    reply_to_message_id: str = "",
) -> Message:
    """Create a plain Message carrying compact JSON payload."""
    return make_message(
        metadata=_metadata(
            message_type=message_type,
            reply_to_message_id=reply_to_message_id,
        ),
        content=RecordDict(
            {
                "payload": ConfigRecord(
                    {"json": json.dumps(payload, separators=(",", ":"))}
                )
            }
        ),
    )


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
    assert request.metadata.message_type == "query"
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
    response_payload: JSONObject = {
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
    assert response.metadata.message_type == "query"
    assert response.metadata.dst_task_id == 456
    assert response.metadata.reply_to_message_id == "request-message-id"
    assert response.payload == response_payload
    assert "response" not in response.payload


def test_model_request_from_message_wraps_plain_message() -> None:
    """ModelRequest should parse a plain Message carrying a request payload."""
    message = _message_with_payload(
        {
            "model": "gpt-5",
            "input": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
        message_type=ModelRequest.MESSAGE_TYPE,
    )

    parsed = ModelRequest.from_message(message)
    assert isinstance(parsed, ModelRequest)
    assert parsed.payload == {
        "model": "gpt-5",
        "input": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }
    assert parsed.metadata.dst_task_id == 123


def test_model_response_from_message_wraps_plain_message() -> None:
    """ModelResponse should parse a plain Message carrying a response payload."""
    message = _message_with_payload(
        {"object": "response", "id": "resp_123"},
        message_type=ModelResponse.MESSAGE_TYPE,
        reply_to_message_id="request-message-id",
    )

    parsed = ModelResponse.from_message(message)
    assert isinstance(parsed, ModelResponse)
    assert parsed.payload == {"object": "response", "id": "resp_123"}
    assert parsed.metadata.reply_to_message_id == "request-message-id"


def test_model_request_from_message_rejects_wrong_message_type() -> None:
    """ModelRequest parsing should reject non-request messages."""
    message = _message_with_payload(
        {"model": "gpt-5", "input": [], "stream": True},
        message_type="train",
    )

    with pytest.raises(ValueError, match="Expected message type"):
        ModelRequest.from_message(message)


def test_model_response_requires_reply_to_message_id() -> None:
    """ModelResponse should identify the request message it replies to."""
    with pytest.raises(ValueError, match="requires reply_to_message_id"):
        ModelResponse(
            dst_task_id=456,
            response={"object": "response"},
            reply_to_message_id="",
        )


def test_model_response_from_message_requires_reply_to_message_id() -> None:
    """ModelResponse parsing should reject messages without reply metadata."""
    message = _message_with_payload(
        {"object": "response"},
        message_type=ModelResponse.MESSAGE_TYPE,
    )

    with pytest.raises(ValueError, match="requires reply_to_message_id"):
        ModelResponse.from_message(message)


def test_model_request_from_message_rejects_missing_payload() -> None:
    """ModelRequest parsing should require a payload record."""
    message = make_message(
        metadata=_metadata(message_type=ModelRequest.MESSAGE_TYPE),
        content=RecordDict(),
    )

    with pytest.raises(ValueError):
        ModelRequest.from_message(message)


def test_model_request_from_message_rejects_invalid_request_payload() -> None:
    """ModelRequest parsing should validate the minimal request shape."""
    with pytest.raises(ValueError):
        ModelRequest.from_message(
            _message_with_payload(
                {"input": [], "stream": True},
                message_type=ModelRequest.MESSAGE_TYPE,
            )
        )


def test_model_response_from_message_rejects_invalid_response_payload() -> None:
    """ModelResponse parsing should validate the minimal response shape."""
    with pytest.raises(ValueError):
        ModelResponse.from_message(
            _message_with_payload(
                {"object": "chat.completion"},
                message_type=ModelResponse.MESSAGE_TYPE,
                reply_to_message_id="request-message-id",
            )
        )


def test_model_request_rejects_non_json_payload_values() -> None:
    """ModelRequest should reject values that cannot be encoded as JSON."""
    input_with_non_json_value: Any = [{"role": "user", "content": object()}]

    with pytest.raises(ValueError, match="JSON serializable"):
        ModelRequest(
            dst_task_id=123,
            input=input_with_non_json_value,
            model="gpt-5",
            stream=True,
        )
