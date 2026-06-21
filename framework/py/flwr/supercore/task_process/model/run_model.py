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
"""Flower model task process."""


from __future__ import annotations

from logging import DEBUG, ERROR

import grpc

from flwr.common.constant import SubStatus
from flwr.common.logger import log
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    PullTaskInputRequest,
    PullTaskInputResponse,
    PushTaskOutputRequest,
)
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub
from flwr.supercore.app_utils import start_parent_process_monitor
from flwr.supercore.exit import ExitCode, flwr_exit, register_signal_handlers
from flwr.supercore.grpc import create_channel, on_channel_state_change
from flwr.supercore.heartbeat import HeartbeatSender, make_task_heartbeat_fn_grpc
from flwr.supercore.interceptors import (
    AppIoTokenClientInterceptor,
    RuntimeVersionClientInterceptor,
)
from flwr.supercore.retry import RetryInvoker, make_simple_grpc_retry_invoker, wrap_stub
from flwr.supercore.telemetry import EventType, event

from .task import handle_task


def run_model(  # pylint: disable=too-many-locals
    serverappio_api_address: str,
    token: str,
    insecure: bool,
    certificates: bytes | None = None,
    parent_pid: int | None = None,
) -> None:
    """Run Flower model task process."""
    # Monitor the main process in case of SIGKILL
    if parent_pid is not None:
        start_parent_process_monitor(parent_pid)

    channel, stub, retry_invoker = _create_serverappio_stub(
        serverappio_api_address=serverappio_api_address,
        token=token,
        insecure=insecure,
        certificates=certificates,
    )

    # Initialize variables for exit handler
    heartbeat_sender = None
    sub_status = SubStatus.FAILED
    details = "Model task failed with unknown error."
    exit_code = ExitCode.SUCCESS

    def on_exit() -> None:
        log(DEBUG, "[flwr-model] Will push Model task output")

        # Set Grpc max retries to 1 to avoid blocking on exit
        retry_invoker.max_tries = 1

        # Push final status
        pushoutput_req = PushTaskOutputRequest(
            sub_status=sub_status,
            details=details,
        )
        try:
            stub.PushTaskOutput(pushoutput_req)
        except grpc.RpcError as err:
            log(ERROR, "Failed to push task output: %s", str(err))

        # Stop heartbeat sender
        if heartbeat_sender and heartbeat_sender.is_running:
            heartbeat_sender.stop()

        # Close the Grpc connection
        channel.close()

    register_signal_handlers(
        event_type=EventType.FLWR_MODEL_RUN_LEAVE,
        exit_message="Run stopped by user.",
        exit_handlers=[on_exit],
    )

    try:
        # Set up heartbeat sender
        heartbeat_sender = HeartbeatSender(make_task_heartbeat_fn_grpc(stub))
        heartbeat_sender.start()

        # Pull task input from SuperLink
        log(DEBUG, "[flwr-model] Pull task input")
        task_input: PullTaskInputResponse = stub.PullTaskInput(PullTaskInputRequest())

        event(EventType.FLWR_MODEL_RUN_ENTER)

        handle_task(
            stub=stub,
            task_id=task_input.task_id,
            run_id=task_input.run.run_id,
        )

        # Update sub_status and details for successful completion
        sub_status = SubStatus.COMPLETED
        details = ""

    except Exception as ex:  # pylint: disable=broad-exception-caught
        log(ERROR, "`flwr-model` failed", exc_info=ex)

        # Update sub_status and details based on the exception
        sub_status = SubStatus.FAILED
        details = f"Model task failed with exception: {str(ex)}"

        # Set exit code
        exit_code = ExitCode.TASK_PROC_EXCEPTION

    flwr_exit(exit_code, event_type=EventType.FLWR_MODEL_RUN_LEAVE)


def _create_serverappio_stub(
    *,
    serverappio_api_address: str,
    token: str,
    insecure: bool,
    certificates: bytes | None,
) -> tuple[grpc.Channel, ServerAppIoStub, RetryInvoker]:
    """Create a ServerAppIo stub authenticated as the model task."""
    channel = create_channel(
        server_address=serverappio_api_address,
        insecure=insecure,
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
