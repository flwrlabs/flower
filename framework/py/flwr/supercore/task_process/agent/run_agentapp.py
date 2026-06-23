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
"""Flower AgentApp process."""


from logging import DEBUG, ERROR
from pathlib import Path
from queue import Queue

import grpc

from flwr.agentapp import AgentApp, LoadAgentAppError
from flwr.app import Context
from flwr.app.exception import AppExitException
from flwr.cli.config_utils import get_fab_metadata
from flwr.cli.install import install_from_fab
from flwr.cli.utils import get_sha256_hash
from flwr.common.config import (
    get_fused_config_from_dir,
    get_project_config,
    get_project_dir,
)
from flwr.common.constant import RUNTIME_DEPENDENCY_INSTALL, SubStatus
from flwr.common.logger import flush_logs, log, start_log_uploader, stop_log_uploader
from flwr.common.serde import (
    context_from_proto,
    context_to_proto,
    fab_from_proto,
    run_from_proto,
)
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    PullTaskInputRequest,
    PullTaskInputResponse,
    PushTaskOutputRequest,
)
from flwr.supercore.app_utils import start_parent_process_monitor
from flwr.supercore.exit import ExitCode, flwr_exit, register_signal_handlers
from flwr.supercore.heartbeat import HeartbeatSender, make_task_heartbeat_fn_grpc
from flwr.supercore.object_ref import load_app
from flwr.supercore.superexec.dependency_installer import (
    RuntimeDependencyInstallationError,
    cleanup_app_runtime_environment,
    install_app_dependencies,
)
from flwr.supercore.telemetry import EventType, event
from flwr.supercore.typing import JSONObject
from flwr.superlink.grid import GrpcGrid

from .context_items import append_items
from .session import RuntimeAgentResponses, RuntimeAgentSession

_AGENT_INPUT_KEY = "agent.input"


def run_agentapp(  # pylint: disable=R0912, R0913, R0914, R0915, R0917, W0212
    serverappio_api_address: str,
    log_queue: Queue[str | None],
    token: str,
    insecure: bool,
    certificates: bytes | None = None,
    parent_pid: int | None = None,
    runtime_dependency_install: bool = RUNTIME_DEPENDENCY_INSTALL,
) -> None:
    """Run Flower AgentApp process."""
    # Monitor the main process in case of SIGKILL
    if parent_pid is not None:
        start_parent_process_monitor(parent_pid)

    # Initialize the ServerAppIo connection.
    grid = GrpcGrid(
        serverappio_service_address=serverappio_api_address,
        insecure=insecure,
        root_certificates=certificates,
        token=token,
    )

    log_uploader = None
    hash_run_id = None
    sub_status = SubStatus.FAILED
    details = "Task failed with unknown error."
    heartbeat_sender = None
    context: Context | None = None
    runtime_env_dir: Path | None = None
    exit_code = ExitCode.SUCCESS

    def on_exit() -> None:
        log(DEBUG, "[flwr-agentapp] Will push AgentApp task output")

        grid._retry_invoker.max_tries = 1

        if log_uploader:
            flush_logs(log_queue)

        pushoutput_req = PushTaskOutputRequest(
            context=context_to_proto(context) if context else None,
            sub_status=sub_status,
            details=details,
        )
        try:
            grid._stub.PushTaskOutput(pushoutput_req)
        except grpc.RpcError as err:
            log(ERROR, "Failed to push task output: %s", str(err))

        if log_uploader:
            stop_log_uploader(log_queue, log_uploader)

        if heartbeat_sender and heartbeat_sender.is_running:
            heartbeat_sender.stop()

        grid.close()

        cleanup_app_runtime_environment(runtime_env_dir)

    register_signal_handlers(
        event_type=EventType.FLWR_AGENTAPP_RUN_LEAVE,
        exit_message="Task stopped by user.",
        exit_handlers=[on_exit],
    )

    try:
        heartbeat_sender = HeartbeatSender(make_task_heartbeat_fn_grpc(grid._stub))
        heartbeat_sender.start()

        log(DEBUG, "[flwr-agentapp] Pull task input")
        res: PullTaskInputResponse = grid._stub.PullTaskInput(PullTaskInputRequest())

        context = context_from_proto(res.context)
        run = run_from_proto(res.run)
        fab = fab_from_proto(res.fab)
        task_id = res.task_id

        hash_run_id = get_sha256_hash(run.run_id)

        grid.set_run(run)

        log_uploader = start_log_uploader(
            log_queue=log_queue,
            node_id=0,
            run_id=run.run_id,
            stub=grid._stub,
        )

        log(DEBUG, "[flwr-agentapp] Start FAB installation.")
        install_from_fab(fab.content, skip_prompt=True)

        fab_id, fab_version = get_fab_metadata(fab.content)

        app_path = str(get_project_dir(fab_id, fab_version, fab.hash_str))

        if runtime_dependency_install:
            log(DEBUG, "[flwr-agentapp] Installing app dependencies.")
            runtime_env_dir = install_app_dependencies(
                app_path,
                launch_id=token,
                run_id=run.run_id,
                index_context={
                    "component": "agentapp",
                    "project_dir": app_path,
                    "run_id": run.run_id,
                    "launch_id": token,
                    "fab_id": run.fab_id,
                    "fab_version": run.fab_version,
                    "fab_hash": fab.hash_str,
                },
            )
        else:
            log(
                DEBUG,
                "[flwr-agentapp] Runtime dependency installation is disabled.",
            )

        config = get_project_config(app_path)

        agent_app_attr = config["tool"]["flwr"]["app"]["components"]["agentapp"]
        context.run_config = get_fused_config_from_dir(
            Path(app_path), run.override_config
        )

        agent_input = context.run_config.get(_AGENT_INPUT_KEY)
        if agent_input is not None:
            if not isinstance(agent_input, str):
                raise ValueError("context.run_config['agent.input'] must be a string.")
            if agent_input:
                item: JSONObject = {
                    "type": "message",
                    "role": "user",
                    "content": agent_input,
                }
                append_items(context, [item])

        log(
            DEBUG,
            "[flwr-agentapp] Will load AgentApp `%s` in %s",
            agent_app_attr,
            app_path,
        )

        event(
            EventType.FLWR_AGENTAPP_RUN_ENTER,
            event_details={"run-id-hash": hash_run_id},
        )

        responses = RuntimeAgentResponses(
            stub=grid._stub,
            run_id=context.run_id,
            task_id=task_id,
            context=context,
        )
        agent = RuntimeAgentSession(responses=responses)

        # Load and run the AgentApp
        agent_app = load_app(agent_app_attr, LoadAgentAppError, app_path)
        if not isinstance(agent_app, AgentApp):
            raise LoadAgentAppError(
                f"Attribute '{agent_app_attr}' is not of type '{AgentApp.__name__}'.",
            ) from None
        agent_app(agent=agent, context=context)

        # Set sub_status and details for successful completion
        sub_status = SubStatus.COMPLETED
        details = ""

    except Exception as ex:  # pylint: disable=broad-exception-caught
        log(ERROR, "AgentApp raised an exception", exc_info=ex)

        sub_status = SubStatus.FAILED
        details = f"AgentApp failed with exception: {str(ex)}"

        exit_code = ExitCode.TASK_PROC_EXCEPTION
        if isinstance(ex, AppExitException):
            exit_code = ex.exit_code
        elif isinstance(ex, ImportError):
            exit_code = ExitCode.COMMON_APP_IMPORT_ERROR
        elif isinstance(ex, RuntimeDependencyInstallationError):
            exit_code = ExitCode.COMMON_RUNTIME_DEPENDENCY_INSTALLATION_ERROR

    flwr_exit(
        code=exit_code,
        event_type=EventType.FLWR_AGENTAPP_RUN_LEAVE,
        event_details={
            "run-id-hash": hash_run_id,
            "success": exit_code == ExitCode.SUCCESS,
        },
    )
