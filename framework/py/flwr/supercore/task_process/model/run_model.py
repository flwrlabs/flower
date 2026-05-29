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

from logging import DEBUG, ERROR
from pathlib import Path
from queue import Queue
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


def run_model(  # pylint: disable=R0912, R0913, R0914, R0915, R0917
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

    # Initialize variables for exit handler
    log_uploader = None
    heartbeat_sender = None
    runtime_env_dir: Path | None = None
    task_id: int | None = None
    run_id: int | None = None
    request_message: ModelRequest | None = None
    model: str | None = None
    sub_status = SubStatus.FAILED
    details = _UNKNOWN_ERROR_DETAILS
    exit_code = ExitCode.SUCCESS

    def on_exit() -> None:
        if log_uploader:
            stop_log_uploader(log_queue, log_uploader)
        cleanup_app_runtime_environment(runtime_env_dir)

    register_signal_handlers(
        event_type=EventType.FLWR_MODEL_RUN_LEAVE,
        exit_message="Run stopped by user.",
        exit_handlers=[on_exit],
    )

    try:
        _ = runtime_dependency_install

        # Set up heartbeat sender
        heartbeat_sender = HeartbeatSender(make_task_heartbeat_fn_grpc(stub))
        heartbeat_sender.start()

        # Pull task input from SuperLink
        log(DEBUG, "[flwr-model] Pull task input")
        task_input: PullTaskInputResponse = stub.PullTaskInput(PullTaskInputRequest())
        task_id = task_input.task_id
        run_id = run_from_proto(task_input.run).run_id

        # Start log uploader for this run
        log_uploader = start_log_uploader(
            log_queue=log_queue,
            node_id=0,
            run_id=run_id,
            stub=stub,
        )

        # Pull and parse exactly one model request.
        request_message = _pull_single_model_request(stub)
        model_request = _parse_model_request(request_message, task_id)
        model = cast(str, model_request.payload["model"])

        def on_stream_event(event_data: JSONObject) -> None:
            stream_response = _stream_event_response(event_data, model)
            _push_model_response(
                stub,
                request_message,
                task_id,
                run_id,
                stream_response,
            )

        # Invoke the provider and forward stream events to the requesting task.
        provider_response = invoke_model_provider(
            model_request.payload,
            on_stream_event=on_stream_event,
        )
        response = _normalize_response(provider_response)
        _push_model_response(stub, request_message, task_id, run_id, response)

        # Update sub_status and details for successful completion
        sub_status = SubStatus.COMPLETED
        details = ""

    except Exception as ex:  # pylint: disable=broad-exception-caught
        log(ERROR, "`flwr-model` failed", exc_info=ex)

        # Update sub_status and details based on the exception
        sub_status = SubStatus.FAILED
        details = f"Model task failed with exception: {str(ex)}"

        # Set exit code
        exit_code = ExitCode.SERVERAPP_EXCEPTION

        # Push a model error response when request routing metadata is available.
        if (
            task_id is not None
            and run_id is not None
            and request_message is not None
            and request_message.metadata.src_task_id is not None
            and bool(request_message.metadata.message_id)
        ):
            try:
                _push_model_response(
                    stub,
                    request_message,
                    task_id,
                    run_id,
                    _error_response(model, ex),
                )
            except Exception as reply_error:  # pylint: disable=broad-exception-caught
                log(ERROR, "Failed to push model error response: %s", str(reply_error))

    finally:
        log(DEBUG, "[flwr-model] Will push Model task output")

        # Set Grpc max retries to 1 to avoid blocking on exit
        retry_invoker.max_tries = 1

        # Upload any remaining logs before pushing final output
        if log_uploader:
            flush_logs(log_queue)

        # Push final status
        pushoutput_req = PushTaskOutputRequest(
            sub_status=sub_status,
            details=details,
        )
        try:
            stub.PushTaskOutput(pushoutput_req)
        except grpc.RpcError as err:
            log(ERROR, "Failed to push task output: %s", str(err))

        # Stop log uploader for this run and upload final logs
        if log_uploader:
            stop_log_uploader(log_queue, log_uploader)

        # Stop heartbeat sender
        if heartbeat_sender and heartbeat_sender.is_running:
            heartbeat_sender.stop()

        # Close the Grpc connection
        channel.close()

        # Clean up run-scoped runtime environment, if any.
        cleanup_app_runtime_environment(runtime_env_dir)

    flwr_exit(
        exit_code,
        event_type=EventType.FLWR_MODEL_RUN_LEAVE,
        event_details={
            "success": exit_code == ExitCode.SUCCESS,
        },
    )


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


def _push_model_response(
    stub: ServerAppIoStub,
    request_message: ModelRequest,
    src_task_id: int,
    run_id: int,
    response: JSONObject,
) -> None:
    """Push a ModelResponse back to the requesting task."""
    message = ModelResponse(
        dst_task_id=_get_reply_task_id(request_message),
        response=response,
        reply_to_message_id=request_message.metadata.message_id,
    )
    message.metadata.__dict__["_run_id"] = run_id
    message.metadata.src_task_id = src_task_id
    message.metadata.__dict__["_message_id"] = message.object_id
    stub.PushTaskMessage(PushTaskMessageRequest(message=message_to_proto(message)))


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
