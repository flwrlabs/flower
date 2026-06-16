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
"""Connector JSON message tests."""


import json

from flwr.app import ConfigRecord, Message, RecordDict
from flwr.app.message_type import MessageType
from flwr.supercore.corestate.utils_test import create_task_message
from flwr.supercore.json_message.connector_message import (
    ConnectorRequest,
    ConnectorResponse,
)
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


def test_connector_request_builds_and_parses_payload() -> None:
    """ConnectorRequest should carry and parse the connector request payload."""
    name = "web_search"
    call_id = "call_123"
    arguments: JSONObject = {"query": "latest Flower release", "max_results": 5}
    payload: JSONObject = {
        "name": name,
        "call_id": call_id,
        "arguments": arguments,
    }
    request = ConnectorRequest(
        dst_task_id=123,
        name=name,
        call_id=call_id,
        arguments=arguments,
        ttl=10.0,
    )

    assert isinstance(request, Message)
    assert request.metadata.message_type == MessageType.QUERY
    assert request.metadata.dst_task_id == 123
    assert request.metadata.reply_to_message_id == ""
    assert request.metadata.ttl == 10.0
    assert request.payload == payload

    parsed = ConnectorRequest.from_message(
        _message_with_payload(payload, message_type=MessageType.QUERY)
    )
    assert isinstance(parsed, ConnectorRequest)
    assert parsed.payload == payload


def test_connector_response_builds_and_parses_payload() -> None:
    """ConnectorResponse should carry and parse the connector response payload."""
    name = "web_search"
    call_id = "call_123"
    output: JSONObject = {"results": [{"title": "Flower", "url": "https://flower.ai"}]}
    error: JSONObject | None = None
    payload: JSONObject = {
        "name": name,
        "call_id": call_id,
        "output": output,
        "error": error,
    }
    response = ConnectorResponse(
        dst_task_id=456,
        name=name,
        call_id=call_id,
        output=output,
        error=error,
        reply_to_message_id="request-message-id",
    )

    assert isinstance(response, Message)
    assert response.metadata.message_type == MessageType.QUERY
    assert response.metadata.dst_task_id == 456
    assert response.metadata.reply_to_message_id == "request-message-id"
    assert response.payload == payload

    parsed = ConnectorResponse.from_message(
        _message_with_payload(
            payload,
            message_type=MessageType.QUERY,
            reply_to_message_id="request-message-id",
        )
    )
    assert isinstance(parsed, ConnectorResponse)
    assert parsed.payload == payload
