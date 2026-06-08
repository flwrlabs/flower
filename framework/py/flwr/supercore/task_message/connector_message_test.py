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
"""Connector task message tests."""


import json
import re
from collections.abc import Callable

import pytest

from flwr.app import ConfigRecord, Message, RecordDict
from flwr.app.message_type import MessageType
from flwr.common.constant import SUPERLINK_NODE_ID
from flwr.supercore.task_message.connector_message import (
    ConnectorRequest,
    ConnectorResponse,
)
from flwr.supercore.corestate.utils_test import create_task_message
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


def test_connector_messages_create_payloads() -> None:
    """Connector messages should carry connector request and response payloads."""
    request = ConnectorRequest(
        dst_task_id=123,
        name="web_search",
        call_id="call_123",
        arguments={"query": "latest Flower release", "max_results": 5},
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
        "name": "web_search",
        "call_id": "call_123",
        "arguments": {"query": "latest Flower release", "max_results": 5},
    }

    response = ConnectorResponse(
        dst_task_id=456,
        name="web_search",
        call_id="call_123",
        output={"results": [{"title": "Flower", "url": "https://flower.ai"}]},
        error=None,
        reply_to_message_id="request-message-id",
    )

    assert isinstance(response, Message)
    assert response.metadata.message_type == "query"
    assert response.metadata.src_node_id == SUPERLINK_NODE_ID
    assert response.metadata.dst_node_id == SUPERLINK_NODE_ID
    assert response.metadata.dst_task_id == 456
    assert response.metadata.reply_to_message_id == "request-message-id"
    assert response.payload == {
        "name": "web_search",
        "call_id": "call_123",
        "output": {"results": [{"title": "Flower", "url": "https://flower.ai"}]},
        "error": None,
    }


@pytest.mark.parametrize(
    ("parser", "message", "expected_cls", "expected_payload"),
    [
        (
            ConnectorRequest.from_message,
            _message_with_payload(
                {
                    "name": "web_search",
                    "call_id": "call_123",
                    "arguments": {"query": "latest Flower release"},
                },
                message_type=MessageType.QUERY,
            ),
            ConnectorRequest,
            {
                "name": "web_search",
                "call_id": "call_123",
                "arguments": {"query": "latest Flower release"},
            },
        ),
        (
            ConnectorResponse.from_message,
            _message_with_payload(
                {
                    "name": "web_search",
                    "call_id": "call_123",
                    "output": {"answer": "Flower"},
                    "error": None,
                },
                message_type=MessageType.QUERY,
                reply_to_message_id="request-message-id",
            ),
            ConnectorResponse,
            {
                "name": "web_search",
                "call_id": "call_123",
                "output": {"answer": "Flower"},
                "error": None,
            },
        ),
    ],
)
def test_from_message_wraps_plain_message(
    parser: Callable[[Message], ConnectorRequest | ConnectorResponse],
    message: Message,
    expected_cls: type[object],
    expected_payload: JSONObject,
) -> None:
    """Connector messages should parse plain Messages carrying payloads."""
    parsed = parser(message)
    assert isinstance(parsed, expected_cls)
    assert parsed.payload == expected_payload


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (
            lambda: ConnectorResponse(
                dst_task_id=456,
                name="web_search",
                call_id="call_123",
                output={},
                error=None,
                reply_to_message_id="",
            ),
            "reply_to_message_id",
        ),
        (
            lambda: ConnectorRequest.from_message(
                _message_with_payload(
                    {"name": "web_search", "call_id": "call_123", "arguments": {}},
                    message_type="train",
                )
            ),
            "Expected message type",
        ),
        (
            lambda: ConnectorResponse.from_message(
                _message_with_payload(
                    {
                        "name": "web_search",
                        "call_id": "call_123",
                        "output": {},
                        "error": None,
                    },
                    message_type=MessageType.QUERY,
                )
            ),
            "reply_to_message_id",
        ),
        (
            lambda: ConnectorResponse.from_message(
                _message_with_payload(
                    '{"name":"web_search","call_id":"call_123",'
                    '"output":{"score":Infinity},"error":null}',
                    message_type=MessageType.QUERY,
                    reply_to_message_id="request-message-id",
                )
            ),
            "malformed",
        ),
    ],
)
def test_invalid_connector_messages_raise(
    build: Callable[[], object],
    match: str,
) -> None:
    """Connector messages should reject invalid public inputs."""
    with pytest.raises(ValueError, match=match):
        build()


@pytest.mark.parametrize(
    ("build", "expected_message"),
    [
        (
            lambda: ConnectorRequest.from_message(
                _message_with_payload("[]", message_type=MessageType.QUERY)
            ),
            "Payload JSON must be a JSON object.",
        ),
        (
            lambda: ConnectorRequest.from_message(
                _message_with_payload(
                    {"call_id": "call_123", "arguments": {}},
                    message_type=MessageType.QUERY,
                )
            ),
            "ConnectorRequest payload requires a non-empty string field 'name'.",
        ),
        (
            lambda: ConnectorRequest.from_message(
                _message_with_payload(
                    {"name": "web_search", "call_id": "call_123", "arguments": []},
                    message_type=MessageType.QUERY,
                )
            ),
            "ConnectorRequest payload requires a JSON object field 'arguments'.",
        ),
        (
            lambda: ConnectorRequest.from_message(
                _message_with_payload(
                    {"name": "web_search", "arguments": {}},
                    message_type=MessageType.QUERY,
                )
            ),
            "ConnectorRequest payload requires a non-empty string field 'call_id'.",
        ),
        (
            lambda: ConnectorResponse.from_message(
                _message_with_payload(
                    {"name": "web_search", "call_id": "call_123", "error": None},
                    message_type=MessageType.QUERY,
                    reply_to_message_id="request-message-id",
                )
            ),
            "ConnectorResponse payload requires field 'output'.",
        ),
        (
            lambda: ConnectorResponse.from_message(
                _message_with_payload(
                    {"name": "web_search", "call_id": "call_123", "output": {}},
                    message_type=MessageType.QUERY,
                    reply_to_message_id="request-message-id",
                )
            ),
            "ConnectorResponse payload requires field 'error'.",
        ),
        (
            lambda: ConnectorResponse.from_message(
                _message_with_payload(
                    {
                        "name": "web_search",
                        "call_id": "call_123",
                        "output": None,
                        "error": "failed",
                    },
                    message_type=MessageType.QUERY,
                    reply_to_message_id="request-message-id",
                )
            ),
            "ConnectorResponse payload field 'error' must be a JSON object.",
        ),
        (
            lambda: ConnectorResponse.from_message(
                _message_with_payload(
                    {
                        "name": "web_search",
                        "call_id": "call_123",
                        "output": {"answer": "Flower"},
                        "error": {"code": "failed"},
                    },
                    message_type=MessageType.QUERY,
                    reply_to_message_id="request-message-id",
                )
            ),
            "ConnectorResponse payload field 'output' must be null when "
            "'error' is set.",
        ),
    ],
)
def test_invalid_connector_messages_describe_expected_json_shapes(
    build: Callable[[], object],
    expected_message: str,
) -> None:
    """Validation errors should name the expected JSON shape exactly."""
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        build()
