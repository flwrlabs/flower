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

from typing import cast

from flwr.common.retry_invoker import RetryInvoker, constant
from flwr.common.serde import message_from_proto, message_to_proto
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    PullTaskMessageRequest,
    PushTaskMessageRequest,
)
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub
from flwr.supercore.model_message import ModelRequest, ModelResponse
from flwr.supercore.typing import JSONObject

from .provider import ModelProviderError, invoke_model_provider


class _ModelRequestUnavailable(Exception):
    """No model request is available yet."""


def handle_task(stub: ServerAppIoStub, task_id: int, run_id: int) -> None:
    """Run one model task request."""
    request_message = _pull_model_request(stub)

    if request_message.metadata.src_task_id is None:
        raise RuntimeError("Model request source task is not set.")

    model_request = ModelRequest.from_message(request_message)

    def on_stream_event(event_data: JSONObject) -> None:
        _push_model_response(
            stub,
            request_message,
            task_id,
            run_id,
            event_data,
        )

    try:
        provider_response = invoke_model_provider(
            model_request.payload,
            on_stream_event=on_stream_event,
        )
    except Exception as ex:
        error_code = (
            "model_provider_error"
            if isinstance(ex, ModelProviderError)
            else "internal_error"
        )

        _push_model_response(
            stub,
            request_message,
            task_id,
            run_id,
            {
                "type": "response.failed",
                "response": {
                    "object": "response",
                    "status": "failed",
                    "error": {
                        "code": error_code,
                        "message": str(ex),
                    },
                    "output": [],
                },
            },
        )
        raise

    if provider_response is not None:
        _push_model_response(
            stub,
            request_message,
            task_id,
            run_id,
            provider_response,
        )


def _pull_model_request(stub: ServerAppIoStub) -> ModelRequest:
    """Pull one model request, waiting until it becomes available."""

    def pull_model_request_once(stub: ServerAppIoStub) -> ModelRequest:
        """Pull one model request without retrying."""
        pull_response = stub.PullTaskMessage(PullTaskMessageRequest(limit=1))
        messages = [message_from_proto(message) for message in pull_response.messages]
        if not messages:
            raise _ModelRequestUnavailable
        return cast(ModelRequest, messages[0])

    return cast(
        ModelRequest,
        RetryInvoker(
            wait_gen_factory=lambda: constant(1.0),
            recoverable_exceptions=_ModelRequestUnavailable,
            max_tries=None,
            max_time=None,
            jitter=None,
        ).invoke(pull_model_request_once, stub),
    )


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
