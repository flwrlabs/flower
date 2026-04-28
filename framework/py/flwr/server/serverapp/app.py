# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Flower ServerApp process."""


import argparse
from logging import DEBUG, ERROR, INFO
from pathlib import Path
from queue import Queue

import grpc

from flwr.app import RecordDict
from flwr.app.exception import AppExitException
from flwr.cli.config_utils import get_fab_metadata
from flwr.cli.install import install_from_fab
from flwr.cli.utils import get_sha256_hash
from flwr.common.args import add_args_flwr_app_common
from flwr.common.config import (
    get_fused_config_from_dir,
    get_project_config,
    get_project_dir,
)
from flwr.common.constant import (
    RUNTIME_DEPENDENCY_INSTALL,
    SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS,
    Status,
    SubStatus,
)
from flwr.common.exit import ExitCode, flwr_exit, register_signal_handlers
from flwr.common.logger import (
    log,
    mirror_output_to_queue,
    restore_output,
    start_log_uploader,
    stop_log_uploader,
)
from flwr.common.serde import (
    context_from_proto,
    context_to_proto,
    fab_from_proto,
    run_from_proto,
    run_status_to_proto,
)
from flwr.common.telemetry import EventType, event
from flwr.common.typing import RunNotRunningException, RunStatus
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    PullAppInputsRequest,
    PullAppInputsResponse,
    PushAppOutputsRequest,
)
from flwr.proto.run_pb2 import UpdateRunStatusRequest  # pylint: disable=E0611
from flwr.server.grid.grpc_grid import GrpcGrid
from flwr.server.run_serverapp import run as run_
from flwr.supercore.app_utils import start_parent_process_monitor
from flwr.supercore.heartbeat import HeartbeatSender, make_app_heartbeat_fn_grpc
from flwr.supercore.superexec.dependency_installer import (
    cleanup_app_runtime_environment,
    install_app_dependencies,
)


def flwr_serverapp() -> None:
    """Run process-isolated Flower ServerApp."""
    args = _parse_args_run_flwr_serverapp().parse_args()

    if not args.insecure:
        flwr_exit(
            ExitCode.COMMON_TLS_NOT_SUPPORTED,
            "`flwr-serverapp` does not support TLS yet.",
        )

    # Capture stdout/stderr
    log_queue: Queue[str | None] = Queue()
    mirror_output_to_queue(log_queue)

    log(INFO, "Start `flwr-serverapp` process")
    log(
        DEBUG,
        "`flwr-serverapp` will attempt to connect to SuperLink's "
        "ServerAppIo API at %s",
        args.serverappio_api_address,
    )
    run_serverapp(
        serverappio_api_address=args.serverappio_api_address,
        log_queue=log_queue,
        token=args.token,
        certificates=None,
        parent_pid=args.parent_pid,
        runtime_dependency_install=args.runtime_dependency_install,
    )

    # Restore stdout/stderr
    restore_output()


def run_serverapp(  # pylint: disable=R0913, R0914, R0915, R0917, W0212
    serverappio_api_address: str,
    log_queue: Queue[str | None],
    token: str,
    certificates: bytes | None = None,
    parent_pid: int | None = None,
    runtime_dependency_install: bool = RUNTIME_DEPENDENCY_INSTALL,
) -> None:
    """Run Flower ServerApp process."""
    # Monitor the main process in case of SIGKILL
    if parent_pid is not None:
        start_parent_process_monitor(parent_pid)

    # Initialize variables for exit handler
    log_uploader = None
    hash_run_id = None
    run = None
    run_status = None
    heartbeat_sender = None
    grid = None
    context = None
    runtime_env_dir: Path | None = None
    exit_code = ExitCode.SUCCESS

    def on_exit() -> None:
        # Set Grpc max retries to 1 to avoid blocking on exit
        if grid:
            grid._retry_invoker.max_tries = 1

        # Stop heartbeat sender
        if heartbeat_sender and heartbeat_sender.is_running:
            heartbeat_sender.stop()

        # Stop log uploader for this run and upload final logs
        if log_uploader:
            stop_log_uploader(log_queue, log_uploader)

        # Update run status
        if run and run_status and grid:
            try:
                req = UpdateRunStatusRequest(
                    run_id=run.run_id, run_status=run_status_to_proto(run_status)
                )
                grid._stub.UpdateRunStatus(req)
            except grpc.RpcError:
                pass

        # Close the Grpc connection
        if grid:
            grid.close()

        # Clean up run-scoped runtime environment, if any.
        cleanup_app_runtime_environment(runtime_env_dir)

    # Register signal handlers for graceful shutdown
    register_signal_handlers(
        event_type=EventType.FLWR_SERVERAPP_RUN_LEAVE,
        exit_message="Run stopped by user.",
        exit_handlers=[on_exit],
    )

    try:

        # Initialize the GrpcGrid
        grid = GrpcGrid(
            serverappio_service_address=serverappio_api_address,
            root_certificates=certificates,
            token=token,
        )

        # Set up heartbeat sender
        heartbeat_sender = HeartbeatSender(
            make_app_heartbeat_fn_grpc(grid._stub, token)
        )
        heartbeat_sender.start()

        # Pull ServerAppInputs from LinkState
        try:
            log(DEBUG, "[flwr-serverapp] Pull ServerAppInputs")
            req = PullAppInputsRequest(token=token)
            res: PullAppInputsResponse = grid._stub.PullAppInputs(req)
        except grpc.RpcError as ex:
            if ex.code() == grpc.StatusCode.FAILED_PRECONDITION:
                raise RuntimeError("Failed to start the run.") from ex
            raise
        context = context_from_proto(res.context)
        run = run_from_proto(res.run)
        fab = fab_from_proto(res.fab)

        hash_run_id = get_sha256_hash(run.run_id)

        grid.set_run(run)

        # Start log uploader for this run
        log_uploader = start_log_uploader(
            log_queue=log_queue,
            node_id=0,
            run_id=run.run_id,
            stub=grid._stub,
        )

        log(DEBUG, "[flwr-serverapp] Start FAB installation.")
        install_from_fab(fab.content, skip_prompt=True)

        fab_id, fab_version = get_fab_metadata(fab.content)

        app_path = str(get_project_dir(fab_id, fab_version, fab.hash_str))

        if runtime_dependency_install:
            log(DEBUG, "[flwr-serverapp] Installing app dependencies.")
            runtime_env_dir = install_app_dependencies(
                app_path,
                launch_id=token,
                run_id=run.run_id,
                index_context={
                    "component": "serverapp",
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
                "[flwr-serverapp] Runtime dependency installation is disabled.",
            )

        config = get_project_config(app_path)

        # Obtain server app reference and the run config
        server_app_attr = config["tool"]["flwr"]["app"]["components"]["serverapp"]
        server_app_run_config = get_fused_config_from_dir(
            Path(app_path), run.override_config
        )

        # Update run_config in context
        context.run_config = server_app_run_config

        log(
            DEBUG,
            "[flwr-serverapp] Will load ServerApp `%s` in %s",
            server_app_attr,
            app_path,
        )

        event(
            EventType.FLWR_SERVERAPP_RUN_ENTER,
            event_details={"run-id-hash": hash_run_id},
        )

        # Load and run the ServerApp with the Grid
        updated_context = run_(
            grid=grid,
            server_app_dir=app_path,
            server_app_attr=server_app_attr,
            context=context,
        )

        # Send resulting context
        # Temporarily disable pushing resulting context to servicer
        updated_context.state = RecordDict()
        context_proto = context_to_proto(updated_context)
        log(DEBUG, "[flwr-serverapp] Will push ServerAppOutputs")
        out_req = PushAppOutputsRequest(
            token=token, run_id=run.run_id, context=context_proto
        )
        _ = grid._stub.PushAppOutputs(out_req)

        run_status = RunStatus(Status.FINISHED, SubStatus.COMPLETED, "")

    # Raised when the run is already stopped by the user
    except RunNotRunningException:
        log(INFO, "")
        log(INFO, "Run ID %s stopped.", run.run_id)  # type: ignore[union-attr]
        log(INFO, "")
        run_status = None
        # No need to update the exit code since this is expected behavior

    except RuntimeError:
        log(ERROR, "Failed to start run.")
        exit_code = ExitCode.SERVERAPP_RUN_START_REJECTED

    except Exception as ex:  # pylint: disable=broad-exception-caught
        exc_entity = "ServerApp"
        log(ERROR, "%s raised an exception", exc_entity, exc_info=ex)
        run_status = RunStatus(Status.FINISHED, SubStatus.FAILED, str(ex))

        # Set exit code
        exit_code = ExitCode.SERVERAPP_EXCEPTION  # General exit code
        if isinstance(ex, AppExitException):
            exit_code = ex.exit_code

    flwr_exit(
        code=exit_code,
        event_type=EventType.FLWR_SERVERAPP_RUN_LEAVE,
        event_details={
            "run-id-hash": hash_run_id,
            "success": exit_code == ExitCode.SUCCESS,
        },
    )


def _parse_args_run_flwr_serverapp() -> argparse.ArgumentParser:
    """Parse flwr-serverapp command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a Flower ServerApp",
    )
    parser.add_argument(
        "--serverappio-api-address",
        default=SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS,
        type=str,
        help="Address of SuperLink's ServerAppIo API (IPv4, IPv6, or a domain name)."
        f"By default, it is set to {SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS}.",
    )
    add_args_flwr_app_common(parser=parser)
    return parser
