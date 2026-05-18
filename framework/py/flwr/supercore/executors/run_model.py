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
"""Flower ModelApp process."""


from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from logging import ERROR
from pathlib import Path
from queue import Queue
from typing import Any, Protocol, cast

import grpc

from flwr.common import EventType, Message
from flwr.common.constant import RUNTIME_DEPENDENCY_INSTALL, SubStatus
from flwr.common.exit import ExitCode, flwr_exit, register_signal_handlers
from flwr.common.grpc import create_channel, on_channel_state_change
from flwr.common.logger import log, stop_log_uploader
from flwr.common.retry_invoker import make_simple_grpc_retry_invoker, wrap_stub
from flwr.common.serde import message_from_proto, message_to_proto
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    PullTaskInputRequest,
    PullTaskMessageRequest,
    PushRunEventsRequest,
    PushTaskMessageRequest,
    PushTaskOutputRequest,
    RunEventPayload,
)
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub
from flwr.supercore.app_utils import start_parent_process_monitor
from flwr.supercore.executors.model_provider import (
    ModelProviderError,
    ModelProviderResult,
    invoke_responses_model,
)
from flwr.supercore.heartbeat import HeartbeatSender, make_task_heartbeat_fn_grpc
from flwr.supercore.interceptors import (
    AppIoTokenClientInterceptor,
    RuntimeVersionClientInterceptor,
)
from flwr.supercore.superexec.dependency_installer import (
    cleanup_app_runtime_environment,
)
from flwr.supercore.task_message import (
    JsonObject,
    JsonValue,
    ModelTaskMessage,
    ModelTaskResultMessage,
)

MODEL_STARTED_EVENT = "model.started"
MODEL_OUTPUT_DELTA_EVENT = "model.output.delta"
MODEL_COMPLETED_EVENT = "model.completed"
MODEL_FAILED_EVENT = "model.failed"

InvokeModelFn = Callable[
    [JsonObject, Callable[[JsonObject], None] | None], ModelProviderResult
]


class ServerAppIoModelStub(Protocol):
    """Subset of ServerAppIo RPCs used by the model executor."""

    def PullTaskInput(self, request: PullTaskInputRequest) -> Any:
        """Pull task input."""

    def PullTaskMessage(self, request: PullTaskMessageRequest) -> Any:
        """Pull task messages."""

    def PushRunEvents(self, request: PushRunEventsRequest) -> Any:
        """Push run events."""

    def PushTaskMessage(self, request: PushTaskMessageRequest) -> Any:
        """Push task messages."""

    def PushTaskOutput(self, request: PushTaskOutputRequest) -> Any:
        """Push task output."""


@dataclass(frozen=True)
class ModelTaskRequest:
    """Parsed model task request."""

    task_id: int
    message: Message
    spec: ModelTaskMessage
    src_task_id: int


@dataclass(frozen=True)
class ModelTaskReplyTarget:
    """Task message routing fields needed to reply to a model request."""

    task_id: int
    message: Message
    src_task_id: int


def run_model(  # pylint: disable=R0913,R0917
    serverappio_api_address: str,
    log_queue: Queue[str | None],
    token: str,
    certificates: bytes | None = None,
    parent_pid: int | None = None,
    runtime_dependency_install: bool = RUNTIME_DEPENDENCY_INSTALL,
) -> None:
    """Run Flower ModelApp process."""
    if parent_pid is not None:
        start_parent_process_monitor(parent_pid)

    channel = create_channel(
        server_address=serverappio_api_address,
        insecure=certificates is None,
        root_certificates=certificates,
        interceptors=[
            RuntimeVersionClientInterceptor(component_name="flwr-model"),
            AppIoTokenClientInterceptor(token),
        ],
    )
    channel.subscribe(on_channel_state_change)

    heartbeat_sender: HeartbeatSender | None = None
    log_uploader = None
    runtime_env_dir: Path | None = None
    exit_code = ExitCode.SUCCESS
    del runtime_dependency_install

    def on_exit() -> None:
        if heartbeat_sender is not None and heartbeat_sender.is_running:
            heartbeat_sender.stop()
        channel.close()
        if log_uploader:
            stop_log_uploader(log_queue, log_uploader)
        cleanup_app_runtime_environment(runtime_env_dir)

    register_signal_handlers(
        event_type=EventType.FLWR_MODEL_RUN_LEAVE,
        exit_message="Run stopped by user.",
        exit_handlers=[on_exit],
    )

    try:
        stub = ServerAppIoStub(channel)
        wrap_stub(stub, make_simple_grpc_retry_invoker())
        heartbeat_sender = HeartbeatSender(make_task_heartbeat_fn_grpc(stub))
        heartbeat_sender.start()

        _run_model_task(stub, invoke_responses_model)

    except grpc.RpcError as err:
        log(ERROR, "gRPC error occurred: %s", str(err))
        exit_code = ExitCode.CLIENTAPP_COMMUNICATION_ERROR
    except Exception as err:  # pylint: disable=broad-exception-caught
        log(ERROR, "`flwr-model` raised an exception", exc_info=err)
        exit_code = ExitCode.SERVERAPP_EXCEPTION

    flwr_exit(
        code=exit_code,
        event_type=EventType.FLWR_MODEL_RUN_LEAVE,
    )


def _run_model_task(
    stub: ServerAppIoModelStub,
    invoke_model: InvokeModelFn = invoke_responses_model,
) -> None:
    """Run one model task using ServerAppIo task RPCs."""
    task_id = _pull_task_id(stub)
    request: ModelTaskRequest | None = None
    reply_target: ModelTaskReplyTarget | None = None
    sub_status = SubStatus.FAILED
    details = "Model task failed due to unknown reason."

    try:
        reply_target = _pull_model_task_reply_target(stub, task_id)
        request = _parse_model_task_request(reply_target)
        _push_run_event(
            stub,
            MODEL_STARTED_EVENT,
            {"task_id": task_id, "model": request.spec.payload["model"]},
        )

        def on_stream_event(event: JsonObject) -> None:
            _push_run_event(
                stub,
                MODEL_OUTPUT_DELTA_EVENT,
                {
                    "task_id": task_id,
                    "model": request.spec.payload["model"],
                    "event": event,
                },
            )

        provider_result = invoke_model(request.spec.payload, on_stream_event)
        _push_model_result(stub, request, provider_result)
        _push_run_event(
            stub,
            MODEL_COMPLETED_EVENT,
            _model_completed_event_data(task_id, request, provider_result.response),
        )
        sub_status = SubStatus.COMPLETED
        details = ""

    except Exception as err:  # pylint: disable=broad-exception-caught
        error = _error_payload_from_exception(err)
        details = _error_message(error)
        _push_run_event(
            stub, MODEL_FAILED_EVENT, _model_failed_event_data(task_id, error)
        )
        if request is not None:
            _push_model_error_result(stub, request, error)
        elif reply_target is not None:
            _push_model_error_result(stub, reply_target, error)

    finally:
        stub.PushTaskOutput(
            PushTaskOutputRequest(sub_status=sub_status, details=details)
        )


def _pull_task_id(stub: ServerAppIoModelStub) -> int:
    """Pull task input and return the authenticated task ID."""
    response = stub.PullTaskInput(PullTaskInputRequest())
    task_id = int(response.task_id)
    if task_id <= 0:
        raise ValueError("Model task input did not include a valid task_id.")
    return task_id


def _pull_model_task_reply_target(
    stub: ServerAppIoModelStub, task_id: int
) -> ModelTaskReplyTarget:
    """Pull exactly one task message and return its reply routing fields."""
    response = stub.PullTaskMessage(PullTaskMessageRequest(limit=2))
    if len(response.messages) != 1:
        raise ValueError(
            f"Expected exactly one model task message, got {len(response.messages)}."
        )

    message = message_from_proto(response.messages[0])
    src_task_id = message.metadata.src_task_id
    if src_task_id is None or src_task_id <= 0:
        raise ValueError("Model task message must include `src_task_id`.")
    if message.metadata.message_id == "":
        message.metadata.__dict__["_message_id"] = message.object_id
    if message.metadata.dst_task_id != task_id:
        raise ValueError("Model task message is not addressed to this task.")

    return ModelTaskReplyTarget(
        task_id=task_id,
        message=message,
        src_task_id=src_task_id,
    )


def _parse_model_task_request(reply_target: ModelTaskReplyTarget) -> ModelTaskRequest:
    """Parse a pulled task message into a typed model request."""
    spec = ModelTaskMessage.from_message(reply_target.message)
    if spec.dst_task_id != reply_target.task_id:
        raise ValueError("Model task message is not addressed to this task.")

    return ModelTaskRequest(
        task_id=reply_target.task_id,
        message=reply_target.message,
        spec=spec,
        src_task_id=reply_target.src_task_id,
    )


def _push_model_result(
    stub: ServerAppIoModelStub,
    request: ModelTaskRequest,
    provider_result: ModelProviderResult,
) -> None:
    """Push a successful model result task message."""
    response = provider_result.response
    result = ModelTaskResultMessage.create(
        dst_task_id=request.src_task_id,
        response=response,
        response_id=_optional_str(response.get("id")),
        usage=_optional_json_object(response.get("usage")),
        finish_reason=_optional_str(response.get("finish_reason")),
        output=cast(JsonValue, response.get("output")),
        events=provider_result.events or None,
        reply_to_message_id=request.message.metadata.message_id,
    )
    stub.PushTaskMessage(
        PushTaskMessageRequest(message=message_to_proto(result.to_message()))
    )


def _push_model_error_result(
    stub: ServerAppIoModelStub,
    request: ModelTaskRequest | ModelTaskReplyTarget,
    error: JsonObject,
) -> None:
    """Push a failed model result task message."""
    result = ModelTaskResultMessage.create(
        dst_task_id=request.src_task_id,
        response={"status": "failed"},
        error=error,
        reply_to_message_id=request.message.metadata.message_id,
    )
    stub.PushTaskMessage(
        PushTaskMessageRequest(message=message_to_proto(result.to_message()))
    )


def _push_run_event(stub: ServerAppIoModelStub, event: str, data: JsonObject) -> None:
    """Push one compact JSON run event."""
    stub.PushRunEvents(
        PushRunEventsRequest(
            events=[
                RunEventPayload(
                    event=event,
                    data=json.dumps(data, separators=(",", ":"), allow_nan=False),
                )
            ]
        )
    )


def _model_completed_event_data(
    task_id: int, request: ModelTaskRequest, response: JsonObject
) -> JsonObject:
    """Build completed event data."""
    data: JsonObject = {
        "task_id": task_id,
        "model": request.spec.payload["model"],
    }
    if response_id := _optional_str(response.get("id")):
        data["response_id"] = response_id
    return data


def _model_failed_event_data(task_id: int, error: JsonObject) -> JsonObject:
    """Build failed event data."""
    return {"task_id": task_id, "error": error}


def _error_payload_from_exception(err: Exception) -> JsonObject:
    """Normalize an exception into a model result error payload."""
    if isinstance(err, ModelProviderError):
        error = dict(err.error)
        if not isinstance(error.get("message"), str):
            error["message"] = str(err)
        return cast(JsonObject, error)
    return {"type": type(err).__name__, "message": str(err)}


def _error_message(error: JsonObject) -> str:
    """Return the error message to use as task output details."""
    message = error.get("message")
    return message if isinstance(message, str) else "Model task failed."


def _optional_str(value: object) -> str | None:
    """Return value if it is a string."""
    return value if isinstance(value, str) else None


def _optional_json_object(value: object) -> JsonObject | None:
    """Return value if it is a JSON object."""
    return cast(JsonObject, value) if isinstance(value, dict) else None
