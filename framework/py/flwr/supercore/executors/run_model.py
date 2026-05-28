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

from dataclasses import dataclass
from logging import DEBUG, ERROR
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import cast

import grpc

from flwr.common import EventType
from flwr.common.constant import RUNTIME_DEPENDENCY_INSTALL, SubStatus
from flwr.common.exit import ExitCode, flwr_exit, register_signal_handlers
from flwr.common.grpc import create_channel, on_channel_state_change
from flwr.common.logger import (
    flush_logs,
    log,
    start_log_uploader,
    stop_log_uploader,
)
from flwr.common.retry_invoker import (
    RetryInvoker,
    make_simple_grpc_retry_invoker,
    wrap_stub,
)
from flwr.common.serde import message_from_proto, message_to_proto, run_from_proto
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    PullTaskInputRequest,
    PullTaskInputResponse,
    PullTaskMessageRequest,
    PushTaskMessageRequest,
    PushTaskOutputRequest,
)
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub
from flwr.supercore.app_utils import start_parent_process_monitor
from flwr.supercore.heartbeat import HeartbeatSender, make_task_heartbeat_fn_grpc
from flwr.supercore.interceptors import (
    AppIoTokenClientInterceptor,
    RuntimeVersionClientInterceptor,
)
from flwr.supercore.model_message import ModelRequest, ModelResponse
from flwr.supercore.superexec.dependency_installer import (
    cleanup_app_runtime_environment,
)
from flwr.supercore.typing import JSONObject, JSONValue

from .model.provider import ModelProviderError, invoke_model_provider

_UNKNOWN_ERROR_DETAILS = "Model task failed with unknown error."


@dataclass
# pylint: disable-next=too-many-instance-attributes
class _ModelTaskContext:
    """State carried across the model executor task lifecycle."""

    channel: grpc.Channel
    stub: ServerAppIoStub
    retry_invoker: RetryInvoker
    log_queue: Queue[str | None]
    log_uploader: Thread | None = None
    heartbeat_sender: HeartbeatSender | None = None
    runtime_env_dir: Path | None = None
    task_id: int | None = None
    run_id: int | None = None
    request_message: ModelRequest | None = None
    model: str | None = None
    sub_status: str = SubStatus.FAILED
    details: str = _UNKNOWN_ERROR_DETAILS
    exit_code: int = ExitCode.SUCCESS

    def stop_log_uploader(self) -> None:
        """Stop the log uploader if it is running."""
        if self.log_uploader is not None:
            stop_log_uploader(self.log_queue, self.log_uploader)
            self.log_uploader = None

    def cleanup_runtime_environment(self) -> None:
        """Clean up the task runtime environment."""
        cleanup_app_runtime_environment(self.runtime_env_dir)
        self.runtime_env_dir = None


@dataclass(frozen=True)
class _ModelReplyContext:
    """Metadata needed to reply to the requesting task."""

    stub: ServerAppIoStub
    dst_task_id: int
    src_task_id: int
    run_id: int
    reply_to_message_id: str


def run_model(  # pylint: disable=R0913, R0917
    serverappio_api_address: str,
    log_queue: Queue[str | None],
    token: str,
    certificates: bytes | None = None,
    parent_pid: int | None = None,
    runtime_dependency_install: bool = RUNTIME_DEPENDENCY_INSTALL,
) -> None:
    """Run Flower ModelApp process.

    The model executor processes one task-routed model request, replies to the
    requesting task, and then finishes.
    """
    # Monitor the main process in case of SIGKILL
    if parent_pid is not None:
        start_parent_process_monitor(parent_pid)

    channel, stub, retry_invoker = _create_serverappio_stub(
        serverappio_api_address=serverappio_api_address,
        token=token,
        certificates=certificates,
    )
    context = _ModelTaskContext(
        channel=channel,
        stub=stub,
        retry_invoker=retry_invoker,
        log_queue=log_queue,
    )

    def on_exit() -> None:
        context.stop_log_uploader()
        context.cleanup_runtime_environment()

    register_signal_handlers(
        event_type=EventType.FLWR_MODEL_RUN_LEAVE,
        exit_message="Run stopped by user.",
        exit_handlers=[on_exit],
    )

    try:
        _run_model_task(context, runtime_dependency_install)

    except Exception as ex:  # pylint: disable=broad-exception-caught
        _handle_model_task_error(context, ex)

    finally:
        _finish_model_task(context)

    flwr_exit(
        context.exit_code,
        event_type=EventType.FLWR_MODEL_RUN_LEAVE,
        event_details={"success": context.exit_code == ExitCode.SUCCESS},
    )


def _run_model_task(
    context: _ModelTaskContext,
    runtime_dependency_install: bool,
) -> None:
    """Run the successful model task path."""
    _ = runtime_dependency_install
    context.heartbeat_sender = HeartbeatSender(
        make_task_heartbeat_fn_grpc(context.stub)
    )
    context.heartbeat_sender.start()

    log(DEBUG, "[flwr-model] Pull task input")
    task_input: PullTaskInputResponse = context.stub.PullTaskInput(
        PullTaskInputRequest()
    )
    task_id = task_input.task_id
    run_id = run_from_proto(task_input.run).run_id
    context.task_id = task_id
    context.run_id = run_id

    context.log_uploader = start_log_uploader(
        log_queue=context.log_queue,
        node_id=0,
        run_id=run_id,
        stub=context.stub,
    )

    request_message = _pull_single_model_request(context.stub)
    context.request_message = request_message
    model_request = _parse_model_request(request_message, task_id)
    model = cast(str, model_request.payload["model"])
    context.model = model

    provider_response = _invoke_provider_with_events(
        context,
        model_request,
    )
    response = _normalize_response(provider_response)
    _push_model_response(_reply_context(context), response)
    context.sub_status = SubStatus.COMPLETED
    context.details = ""


def _invoke_provider_with_events(
    context: _ModelTaskContext,
    model_request: ModelRequest,
) -> JSONObject:
    """Invoke the provider and forward stream events to the requesting task."""

    def on_stream_event(event: JSONObject) -> None:
        _push_model_stream_event(
            _reply_context(context),
            event,
            model=_require_model(context),
        )

    return invoke_model_provider(
        model_request.payload,
        on_stream_event=on_stream_event,
    )


def _handle_model_task_error(context: _ModelTaskContext, error: Exception) -> None:
    """Handle task errors and push best-effort failure data."""
    log(ERROR, "`flwr-model` failed", exc_info=error)
    context.exit_code = ExitCode.SERVERAPP_EXCEPTION
    context.sub_status = SubStatus.FAILED
    context.details = f"Model task failed with exception: {str(error)}"

    if _can_reply_to_request(context):
        _try_push_model_error_response(
            _reply_context(context),
            model=context.model,
            error=error,
        )


def _finish_model_task(context: _ModelTaskContext) -> None:
    """Push task output and release local resources."""
    context.retry_invoker.max_tries = 1

    if context.log_uploader is not None:
        flush_logs(context.log_queue)

    try:
        context.stub.PushTaskOutput(
            PushTaskOutputRequest(
                sub_status=context.sub_status,
                details=context.details,
            )
        )
    except grpc.RpcError as err:
        log(ERROR, "Failed to push task output: %s", str(err))

    context.stop_log_uploader()

    if context.heartbeat_sender is not None and context.heartbeat_sender.is_running:
        context.heartbeat_sender.stop()
    context.channel.close()
    context.cleanup_runtime_environment()


def _create_serverappio_stub(
    *,
    serverappio_api_address: str,
    token: str,
    certificates: bytes | None,
) -> tuple[grpc.Channel, ServerAppIoStub, RetryInvoker]:
    """Create a ServerAppIo stub authenticated as the model task."""
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
    stub = ServerAppIoStub(channel)
    retry_invoker = make_simple_grpc_retry_invoker()
    wrap_stub(stub, retry_invoker)
    return channel, stub, retry_invoker


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


def _get_reply_task_id(request_message: ModelRequest) -> int:
    """Return the task ID that should receive the model response."""
    if request_message.metadata.src_task_id is None:
        raise RuntimeError("Model request source task is not set.")
    return request_message.metadata.src_task_id


def _require_task_id(context: _ModelTaskContext) -> int:
    """Return the task ID if the task input has been pulled."""
    if context.task_id is None:
        raise RuntimeError("Model task ID is not set.")
    return context.task_id


def _require_run_id(context: _ModelTaskContext) -> int:
    """Return the run ID if the task input has been pulled."""
    if context.run_id is None:
        raise RuntimeError("Model run ID is not set.")
    return context.run_id


def _require_request_message(context: _ModelTaskContext) -> ModelRequest:
    """Return the request message if it has been pulled."""
    if context.request_message is None:
        raise RuntimeError("Model request message is not set.")
    return context.request_message


def _require_model(context: _ModelTaskContext) -> str:
    """Return the model name if the model request has been parsed."""
    if context.model is None:
        raise RuntimeError("Model name is not set.")
    return context.model


def _can_reply_to_request(context: _ModelTaskContext) -> bool:
    """Return true if enough request metadata is available for a response."""
    if (
        context.task_id is None
        or context.run_id is None
        or context.request_message is None
    ):
        return False
    return context.request_message.metadata.src_task_id is not None and bool(
        context.request_message.metadata.message_id
    )


def _reply_context(context: _ModelTaskContext) -> _ModelReplyContext:
    """Build response routing metadata from the task context."""
    request_message = _require_request_message(context)
    return _ModelReplyContext(
        stub=context.stub,
        dst_task_id=_get_reply_task_id(request_message),
        src_task_id=_require_task_id(context),
        run_id=_require_run_id(context),
        reply_to_message_id=request_message.metadata.message_id,
    )


def _push_model_response(
    reply: _ModelReplyContext,
    response: JSONObject,
) -> None:
    """Push a ModelResponse back to the requesting task."""
    message = ModelResponse(
        dst_task_id=reply.dst_task_id,
        response=response,
        reply_to_message_id=reply.reply_to_message_id,
    )
    message.metadata.__dict__["_run_id"] = reply.run_id
    message.metadata.src_task_id = reply.src_task_id
    message.metadata.__dict__["_message_id"] = message.object_id
    reply.stub.PushTaskMessage(
        PushTaskMessageRequest(message=message_to_proto(message))
    )


def _try_push_model_error_response(
    reply: _ModelReplyContext,
    *,
    model: str | None,
    error: Exception,
) -> None:
    """Best-effort model error response push."""
    try:
        _push_model_response(reply, _error_response(model, error))
    except Exception as reply_error:  # pylint: disable=broad-exception-caught
        log(ERROR, "Failed to push model error response: %s", str(reply_error))


def _push_model_stream_event(
    reply: _ModelReplyContext,
    event: JSONObject,
    *,
    model: str,
) -> None:
    """Push one provider stream event back to the requesting task."""
    _push_model_response(reply, _stream_event_response(event, model))


def _normalize_response(response: JSONObject) -> JSONObject:
    """Ensure provider output can be sent as a ModelResponse payload."""
    normalized = dict(response)
    normalized.setdefault("object", "response")
    return cast(JSONObject, normalized)


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
