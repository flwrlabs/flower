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
from collections.abc import Callable

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
    payload: JSONObject,
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


@pytest.mark.parametrize(
    ("parser", "message", "expected_cls", "expected_payload"),
    [
        (
            ModelRequest.from_message,
            _message_with_payload(
                {
                    "model": "gpt-5",
                    "input": [{"role": "user", "content": "Hello"}],
                    "stream": False,
                },
                message_type=ModelRequest.MESSAGE_TYPE,
            ),
            ModelRequest,
            {
                "model": "gpt-5",
                "input": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        ),
        (
            ModelResponse.from_message,
            _message_with_payload(
                {"object": "response", "id": "resp_123"},
                message_type=ModelResponse.MESSAGE_TYPE,
                reply_to_message_id="request-message-id",
            ),
            ModelResponse,
            {"object": "response", "id": "resp_123"},
        ),
    ],
)
def test_from_message_wraps_plain_message(
    parser: Callable[[Message], ModelRequest | ModelResponse],
    message: Message,
    expected_cls: type[object],
    expected_payload: JSONObject,
) -> None:
    """Model messages should parse plain Messages carrying model payloads."""
    parsed = parser(message)
    assert isinstance(parsed, expected_cls)
    assert parsed.payload == expected_payload


def test_model_response_requires_reply_to_message_id() -> None:
    """ModelResponse should identify the request message it replies to."""
    with pytest.raises(ValueError, match="requires reply_to_message_id"):
        ModelResponse(
            dst_task_id=456,
            response={"object": "response"},
            reply_to_message_id="",
        )


@pytest.mark.parametrize(
    ("parser", "message", "match"),
    [
        (
            ModelRequest.from_message,
            _message_with_payload(
                {"model": "gpt-5", "input": [], "stream": True},
                message_type="train",
            ),
            "Expected message type",
        ),
        (
            ModelRequest.from_message,
            make_message(
                metadata=_metadata(message_type=ModelRequest.MESSAGE_TYPE),
                content=RecordDict(),
            ),
            "payload",
        ),
        (
            ModelRequest.from_message,
            _message_with_payload(
                {"input": [], "stream": True},
                message_type=ModelRequest.MESSAGE_TYPE,
            ),
            "model",
        ),
        (
            ModelResponse.from_message,
            _message_with_payload(
                {"object": "response"},
                message_type=ModelResponse.MESSAGE_TYPE,
            ),
            "reply_to_message_id",
        ),
        (
            ModelResponse.from_message,
            _message_with_payload(
                {"object": "chat.completion"},
                message_type=ModelResponse.MESSAGE_TYPE,
                reply_to_message_id="request-message-id",
            ),
            "Responses object",
        ),
    ],
)
def test_from_message_rejects_invalid_message(
    parser: Callable[[Message], object],
    message: Message,
    match: str,
) -> None:
    """Model message parsers should reject invalid plain Messages."""
    with pytest.raises(ValueError, match=match):
        parser(message)
