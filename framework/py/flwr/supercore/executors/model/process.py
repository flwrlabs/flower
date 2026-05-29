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
"""Model task processing."""


from __future__ import annotations

from typing import cast

from flwr.common.serde import message_from_proto, message_to_proto
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    PullTaskMessageRequest,
    PushTaskMessageRequest,
)
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub
from flwr.supercore.model_message import ModelRequest, ModelResponse
from flwr.supercore.typing import JSONObject, JSONValue

from .provider import ModelProviderError, invoke_model_provider


def execute_model_task(stub: ServerAppIoStub, task_id: int, run_id: int) -> None:
    """Execute one model task request."""
    request_message = _pull_single_model_request(stub)
    model_request = _parse_model_request(request_message, task_id)
    model = cast(str, model_request.payload["model"])

    def on_stream_event(event_data: JSONObject) -> None:
        _push_model_response(
            stub,
            request_message,
            task_id,
            run_id,
            _stream_event_response(event_data, model),
        )

    try:
        provider_response = invoke_model_provider(
            model_request.payload,
            on_stream_event=on_stream_event,
        )
    except Exception as ex:
        _push_model_response(
            stub,
            request_message,
            task_id,
            run_id,
            _error_response(model, ex),
        )
        raise

    response = dict(provider_response)
    response.setdefault("object", "response")
    _push_model_response(
        stub,
        request_message,
        task_id,
        run_id,
        cast(JSONObject, response),
    )


def _pull_single_model_request(stub: ServerAppIoStub) -> ModelRequest:
    """Pull and parse exactly one model request message."""
    response = stub.PullTaskMessage(PullTaskMessageRequest(limit=1))
    messages = [message_from_proto(message) for message in response.messages]
    if len(messages) != 1:
        raise RuntimeError(f"Expected exactly one model request, got {len(messages)}.")
    return cast(ModelRequest, messages[0])


def _parse_model_request(message: ModelRequest, task_id: int) -> ModelRequest:
    """Parse a task message as a model request for this task."""
    if message.metadata.dst_task_id != task_id:
        raise RuntimeError(
            "Model request destination does not match the authenticated task."
        )
    if message.metadata.src_task_id is None:
        raise RuntimeError("Model request source task is not set.")
    if not message.metadata.message_id:
        raise RuntimeError("Model request message ID is not set.")
    return ModelRequest.from_message(message)


def _push_model_response(
    stub: ServerAppIoStub,
    request_message: ModelRequest,
    src_task_id: int,
    run_id: int,
    response: JSONObject,
) -> None:
    """Push a ModelResponse back to the requesting task."""
    if request_message.metadata.src_task_id is None:
        raise RuntimeError("Model request source task is not set.")
    message = ModelResponse(
        dst_task_id=request_message.metadata.src_task_id,
        response=response,
        reply_to_message_id=request_message.metadata.message_id,
    )
    message.metadata.__dict__["_run_id"] = run_id
    message.metadata.src_task_id = src_task_id
    message.metadata.__dict__["_message_id"] = message.object_id
    stub.PushTaskMessage(PushTaskMessageRequest(message=message_to_proto(message)))


def _stream_event_response(event: JSONObject, model: str) -> JSONObject:
    """Wrap one provider stream event in a task-routed model response."""
    data = dict(event)
    event_type = data.get("type")
    if not isinstance(event_type, str) or not event_type:
        data["type"] = "response.event"
    data.setdefault("model", model)
    return {
        "object": "response",
        "status": "in_progress",
        "model": model,
        "events": [cast(JSONObject, data)],
    }


def _error_response(model: str | None, error: Exception) -> JSONObject:
    """Build a Responses-compatible failed response."""
    response: JSONObject = {
        "object": "response",
        "status": "failed",
        "error": _error_payload(error),
    }
    if model is not None:
        response["model"] = model
    return response


def _error_payload(error: Exception) -> JSONObject:
    """Build a structured error payload."""
    payload: JSONObject = {
        "type": error.__class__.__name__,
        "message": str(error),
    }
    if isinstance(error, ModelProviderError):
        payload["provider_status_code"] = error.status_code
        payload["provider_detail"] = cast(JSONValue, error.detail)
    return payload
