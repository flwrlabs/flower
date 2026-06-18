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
"""Handle connector tasks."""

from __future__ import annotations

import time
from typing import cast

from flwr.common.serde import message_from_proto, message_to_proto
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    PullTaskMessageRequest,
    PushTaskMessageRequest,
)
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub
from flwr.supercore.json_message.connector_message import (
    ConnectorRequest,
    ConnectorResponse,
)
from flwr.supercore.typing import JSONObject

from .registry import invoke_connector


def handle_task(
    stub: ServerAppIoStub,
    task_id: int,
    run_id: int,
) -> None:
    """Run one connector task request."""
    request_message = _pull_connector_request(stub)
    if request_message.metadata.src_task_id is None:
        raise RuntimeError("Connector request source task is not set.")

    def _push_connector_response(response: JSONObject) -> None:
        """Push a ConnectorResponse back to the requesting task."""
        message = ConnectorResponse(
            dst_task_id=cast(int, request_message.metadata.src_task_id),
            name=cast(str, request_message.payload["name"]),
            call_id=cast(str, request_message.payload["call_id"]),
            output=response["output"],
            error=cast(JSONObject | None, response["error"]),
            reply_to_message_id=request_message.metadata.message_id,
        )
        message.metadata.__dict__["_run_id"] = run_id
        message.metadata.src_task_id = task_id
        message.metadata.__dict__["_message_id"] = message.object_id
        stub.PushTaskMessage(PushTaskMessageRequest(message=message_to_proto(message)))

    response = None
    try:
        response = {
            "output": invoke_connector(
                name=cast(str, request_message.payload["name"]),
                arguments=cast(JSONObject, request_message.payload["arguments"]),
            ),
            "error": None,
        }
    except Exception as ex:
        response = _make_error_response(ex)
        raise
    finally:
        # Push the response
        if response is not None:
            _push_connector_response(response)


def _pull_connector_request(stub: ServerAppIoStub) -> ConnectorRequest:
    """Pull one connector request, waiting until it becomes available."""
    # Keep polling until flwr-agentapp produces a request. If it exits, cleanup
    # forces flwr-connector to stop, with auth handling revoked tokens.
    while True:
        pull_response = stub.PullTaskMessage(PullTaskMessageRequest(limit=1))
        messages = [message_from_proto(message) for message in pull_response.messages]
        if messages:
            return ConnectorRequest.from_message(messages[0])
        time.sleep(1)  # Wait for 1 second before trying again.


def _make_error_response(ex: Exception) -> JSONObject:
    """Create a JSON error response from an exception."""
    return {
        "output": None,
        "error": {
            "code": "connector_error",
            "message": str(ex),
        },
    }
