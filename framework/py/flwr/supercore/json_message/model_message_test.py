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
"""Model JSON message tests."""


import json
import re
from collections.abc import Callable

import pytest

from flwr.app import ConfigRecord, Message, RecordDict
from flwr.app.message_type import MessageType
from flwr.common.constant import SUPERLINK_NODE_ID
from flwr.supercore.corestate.utils_test import create_task_message
from flwr.supercore.json_message.model_message import ModelRequest, ModelResponse
from flwr.supercore.typing import JSONObject


def _message_with_payload(
    payload: JSONObject | str,
    *,
    message_type: str,
    reply_to_message_id: str = "",
) -> Message:
    """Create a plain Message carrying compact or raw JSON payload."""
    payload_json = (
        payload
        if isinstance(payload, str)
        else json.dumps(payload, separators=(",", ":"))
    )
    return create_task_message(
        content=RecordDict({"payload": ConfigRecord({"json": payload_json})}),
        message_type=message_type,
        reply_to_message_id=reply_to_message_id,
        dst_task_id=123,
    )


def test_model_messages_create_payloads() -> None:
    """Model messages should carry their Responses payloads."""
    request = ModelRequest(
        dst_task_id=123,
        input_=[{"role": "user", "content": "Hello"}],
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
    assert request.metadata.src_node_id == SUPERLINK_NODE_ID
    assert request.metadata.dst_node_id == SUPERLINK_NODE_ID
    assert request.metadata.src_task_id is None
    assert request.metadata.dst_task_id == 123
    assert request.metadata.reply_to_message_id == ""
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
    assert response.metadata.src_node_id == SUPERLINK_NODE_ID
    assert response.metadata.dst_node_id == SUPERLINK_NODE_ID
    assert response.metadata.dst_task_id == 456
    assert response.metadata.reply_to_message_id == "request-message-id"
    assert response.payload == response_payload


def test_model_request_accepts_string_input_and_default_stream() -> None:
    """Model requests should accept simple string prompts."""
    request = ModelRequest(
        dst_task_id=123,
        input_="Hello",
        model="gpt-5",
    )

    assert request.payload == {
        "model": "gpt-5",
        "input": "Hello",
        "stream": False,
    }


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
                message_type=MessageType.QUERY,
            ),
            ModelRequest,
            {
                "model": "gpt-5",
                "input": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        ),
        (
            ModelRequest.from_message,
            _message_with_payload(
                {
                    "model": "gpt-5",
                    "input": "Hello",
                },
                message_type=MessageType.QUERY,
            ),
            ModelRequest,
            {
                "model": "gpt-5",
                "input": "Hello",
            },
        ),
        (
            ModelRequest.from_message,
            _message_with_payload(
                {
                    "model": "gpt-5",
                    "input": [],
                    "stream": True,
                },
                message_type="train",
            ),
            ModelRequest,
            {
                "model": "gpt-5",
                "input": [],
                "stream": True,
            },
        ),
        (
            ModelResponse.from_message,
            _message_with_payload(
                {"object": "response", "id": "resp_123"},
                message_type=MessageType.QUERY,
                reply_to_message_id="request-message-id",
            ),
            ModelResponse,
            {"object": "response", "id": "resp_123"},
        ),
        (
            ModelResponse.from_message,
            _message_with_payload(
                {"object": "response"},
                message_type=MessageType.QUERY,
            ),
            ModelResponse,
            {"object": "response"},
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


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (
            lambda: ModelResponse(
                dst_task_id=456,
                response={"object": "response"},
                reply_to_message_id="",
            ),
            "reply_to_message_id",
        ),
        (
            lambda: ModelRequest.from_message(
                _message_with_payload(
                    {"input": [], "stream": True},
                    message_type=MessageType.QUERY,
                )
            ),
            "model",
        ),
        (
            lambda: ModelRequest.from_message(
                _message_with_payload(
                    {"model": "gpt-5", "input": 1},
                    message_type=MessageType.QUERY,
                )
            ),
            "input",
        ),
        (
            lambda: ModelRequest.from_message(
                _message_with_payload(
                    {"model": "gpt-5", "input": ["Hello"]},
                    message_type=MessageType.QUERY,
                )
            ),
            "input",
        ),
        (
            lambda: ModelRequest.from_message(
                _message_with_payload(
                    {"model": "gpt-5", "input": "Hello", "stream": "false"},
                    message_type=MessageType.QUERY,
                )
            ),
            "stream",
        ),
        (
            lambda: ModelRequest.from_message(
                _message_with_payload(
                    '{"model":"gpt-5","input":"Hello","tool_choice":NaN}',
                    message_type=MessageType.QUERY,
                )
            ),
            "malformed",
        ),
        (
            lambda: ModelResponse.from_message(
                _message_with_payload(
                    '{"object":"response","error":{"code":Infinity}}',
                    message_type=MessageType.QUERY,
                    reply_to_message_id="request-message-id",
                )
            ),
            "malformed",
        ),
    ],
)
def test_invalid_model_messages_raise(
    build: Callable[[], object],
    match: str,
) -> None:
    """Model messages should reject invalid public inputs."""
    with pytest.raises(ValueError, match=match):
        build()


@pytest.mark.parametrize(
    ("build", "expected_message"),
    [
        (
            lambda: ModelRequest.from_message(
                _message_with_payload("[]", message_type=MessageType.QUERY)
            ),
            "Payload JSON must be a JSON object.",
        ),
        (
            lambda: ModelRequest.from_message(
                _message_with_payload(
                    {"model": "gpt-5", "input": 1},
                    message_type=MessageType.QUERY,
                )
            ),
            "ModelRequest payload field 'input' must be a string or sequence "
            "of JSON objects.",
        ),
        (
            lambda: ModelRequest.from_message(
                _message_with_payload(
                    {"model": "gpt-5", "input": "Hello", "tools": ["web"]},
                    message_type=MessageType.QUERY,
                )
            ),
            "ModelRequest payload field 'tools' must be a sequence of JSON objects.",
        ),
        (
            lambda: ModelRequest.from_message(
                _message_with_payload(
                    {"model": "gpt-5", "input": "Hello", "reasoning": []},
                    message_type=MessageType.QUERY,
                )
            ),
            "ModelRequest payload field 'reasoning' must be a JSON object.",
        ),
        (
            lambda: ModelResponse.from_message(
                _message_with_payload(
                    {"object": "response", "output": ["message"]},
                    message_type=MessageType.QUERY,
                    reply_to_message_id="request-message-id",
                )
            ),
            "ModelResponse payload field 'output' must be a sequence of JSON objects.",
        ),
        (
            lambda: ModelResponse.from_message(
                _message_with_payload(
                    {"object": "response", "error": "failed"},
                    message_type=MessageType.QUERY,
                    reply_to_message_id="request-message-id",
                )
            ),
            "ModelResponse payload field 'error' must be a JSON object.",
        ),
        (
            lambda: ModelResponse.from_message(
                _message_with_payload(
                    {"object": "not_response"},
                    message_type=MessageType.QUERY,
                    reply_to_message_id="request-message-id",
                )
            ),
            "ModelResponse payload field 'object' must be 'response'.",
        ),
    ],
)
def test_invalid_model_messages_describe_expected_json_shapes(
    build: Callable[[], object],
    expected_message: str,
) -> None:
    """Validation errors should name the expected JSON shape exactly."""
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        build()
