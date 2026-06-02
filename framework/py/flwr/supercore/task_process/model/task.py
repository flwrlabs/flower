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
"""Handle model tasks."""


from __future__ import annotations

import time
from typing import cast

from flwr.common.serde import message_from_proto, message_to_proto
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    PullTaskMessageRequest,
    PushTaskEventsRequest,
    PushTaskMessageRequest,
)
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub
from flwr.proto.task_pb2 import TaskEvent  # pylint: disable=E0611
from flwr.supercore.model_message import ModelRequest, ModelResponse
from flwr.supercore.typing import JSONObject
from flwr.supercore.utils import strict_json_dumps

from .provider import ModelProviderError, invoke_model_provider

_DEFAULT_TASK_EVENT_BATCH_SIZE = 16


def handle_task(stub: ServerAppIoStub, task_id: int, run_id: int) -> None:
    """Run one model task request."""
    request_message = _pull_model_request(stub)
    if request_message.metadata.src_task_id is None:
        raise RuntimeError("Model request source task is not set.")

    def _push_model_response(response: JSONObject) -> None:
        """Push a ModelResponse back to the requesting task."""
        message = ModelResponse(
            dst_task_id=cast(int, request_message.metadata.src_task_id),
            response=response,
            reply_to_message_id=request_message.metadata.message_id,
        )
        message.metadata.__dict__["_run_id"] = run_id
        message.metadata.src_task_id = task_id
        message.metadata.__dict__["_message_id"] = message.object_id
        stub.PushTaskMessage(PushTaskMessageRequest(message=message_to_proto(message)))

    request_payload = request_message.payload
    if request_payload.get("stream") is True:
        events: list[TaskEvent] = []

        def _flush_events() -> None:
            """Push buffered stream events."""
            if not events:
                return
            batch = list(events)
            events.clear()
            try:
                stub.PushTaskEvents(PushTaskEventsRequest(events=batch))
            except Exception:
                events[:0] = batch
                raise

        def _publish_event(event: JSONObject) -> None:
            """Buffer one Open Responses stream event."""
            encoded = strict_json_dumps(event, compact=True)
            events.append(TaskEvent(event=cast(str, event["type"]), data=encoded))
            if len(events) >= _DEFAULT_TASK_EVENT_BATCH_SIZE:
                _flush_events()

        try:
            invoke_model_provider(
                request_payload,
                on_stream_event=_publish_event,
            )
        finally:
            _flush_events()
        return

    try:
        response = invoke_model_provider(request_payload)
    except Exception as ex:
        response = _make_error_response(ex)
        raise
    finally:
        if response is not None:
            _push_model_response(response)


def _pull_model_request(stub: ServerAppIoStub) -> ModelRequest:
    """Pull one model request, waiting until it becomes available."""
    # Keep polling until flwr-agentapp produces a request. If it exits, cleanup
    # forces flwr-model to stop, with auth handling revoked tokens.
    while True:
        pull_response = stub.PullTaskMessage(PullTaskMessageRequest(limit=1))
        messages = [message_from_proto(message) for message in pull_response.messages]
        if messages:
            return ModelRequest.from_message(messages[0])
        time.sleep(1)  # Wait for 1 second before trying again.


def _make_error_response(ex: Exception) -> JSONObject:
    """Create a JSON error response from an exception."""
    return {
        "object": "response",
        "status": "failed",
        "error": {
            "code": _error_code(ex),
            "message": str(ex),
        },
        "output": [],
    }


def _error_code(ex: Exception) -> str:
    """Return the error code for a model task exception."""
    if isinstance(ex, ModelProviderError):
        return "model_provider_error"
    return "internal_error"
