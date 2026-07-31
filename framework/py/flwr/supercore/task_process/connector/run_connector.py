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
"""Flower connector task process."""


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


def run_connector(  # pylint: disable=too-many-locals
    serverappio_api_address: str,
    token: str,
    insecure: bool,
    certificates: bytes | None = None,
    parent_pid: int | None = None,
) -> None:
    """Run Flower connector task process."""
    if parent_pid is not None:
        start_parent_process_monitor(parent_pid)

    channel, stub, retry_invoker = _create_serverappio_stub(
        serverappio_api_address=serverappio_api_address,
        token=token,
        insecure=insecure,
        certificates=certificates,
    )

    heartbeat_sender = None
    sub_status = SubStatus.FAILED
    details = "Connector task failed with unknown error."
    exit_code = ExitCode.SUCCESS

    def on_exit() -> None:
        log(DEBUG, "[flwr-connector] Will push Connector task output")

        retry_invoker.max_tries = 1

        pushoutput_req = PushTaskOutputRequest(
            sub_status=sub_status,
            details=details,
        )
        try:
            stub.PushTaskOutput(pushoutput_req)
        except grpc.RpcError as err:
            log(ERROR, "Failed to push task output: %s", str(err))

        if heartbeat_sender and heartbeat_sender.is_running:
            heartbeat_sender.stop()

        channel.close()

    register_signal_handlers(
        event_type=EventType.FLWR_CONNECTOR_RUN_LEAVE,
        exit_message="Task stopped by user.",
        exit_handlers=[on_exit],
    )

    try:
        heartbeat_sender = HeartbeatSender(make_task_heartbeat_fn_grpc(stub))
        heartbeat_sender.start()

        log(DEBUG, "[flwr-connector] Pull task input")
        task_input: PullTaskInputResponse = stub.PullTaskInput(PullTaskInputRequest())

        event(EventType.FLWR_CONNECTOR_RUN_ENTER)

        handle_task(
            stub=stub,
            task_id=task_input.task_id,
            run_id=task_input.run.run_id,
        )

        sub_status = SubStatus.COMPLETED
        details = ""

    except Exception as ex:  # pylint: disable=broad-exception-caught
        log(ERROR, "`flwr-connector` failed", exc_info=ex)

        sub_status = SubStatus.FAILED
        details = f"Connector task failed with exception: {str(ex)}"

        exit_code = ExitCode.TASK_PROC_EXCEPTION
        if isinstance(ex, ImportError):
            exit_code = ExitCode.COMMON_APP_IMPORT_ERROR

    flwr_exit(exit_code, event_type=EventType.FLWR_CONNECTOR_RUN_LEAVE)


def _create_serverappio_stub(
    *,
    serverappio_api_address: str,
    token: str,
    insecure: bool,
    certificates: bytes | None,
) -> tuple[grpc.Channel, ServerAppIoStub, RetryInvoker]:
    """Create a ServerAppIo stub authenticated as the connector task."""
    channel = create_channel(
        server_address=serverappio_api_address,
        insecure=insecure,
        root_certificates=certificates,
        interceptors=[
            RuntimeVersionClientInterceptor(component_name="flwr-connector"),
            AppIoTokenClientInterceptor(token),
        ],
    )
    channel.subscribe(on_channel_state_change)
    stub = ServerAppIoStub(channel)
    retry_invoker = make_simple_grpc_retry_invoker()
    wrap_stub(stub, retry_invoker)
    return channel, stub, retry_invoker
