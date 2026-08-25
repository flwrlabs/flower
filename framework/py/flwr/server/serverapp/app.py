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
import json
from logging import DEBUG, ERROR, INFO
from pathlib import Path
from queue import Queue
from typing import Any
import os

import grpc

from flwr.app.exception import AppExitException
from flwr.cli.config_utils import get_fab_metadata
from flwr.cli.install import install_from_fab
from flwr.cli.utils import get_sha256_hash
from flwr.common.args import add_args_flwr_app_common
from flwr.common import ConfigRecord
from flwr.common.config import (
    get_flwr_dir,
    get_fused_config_from_dir,
    get_project_config,
    get_project_dir,
)
from flwr.common.constant import (
    SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS,
    ExecPluginType,
    Status,
    SubStatus,
)
from flwr.common.exit import ExitCode, flwr_exit, register_signal_handlers
from flwr.common.grpc import create_channel
from flwr.common.logger import (
    log,
    mirror_output_to_queue,
    restore_output,
    start_log_uploader,
    stop_log_uploader,
)
from flwr.common.profiling import (
    ProfileRecorder,
    clear_active_profiler,
    clear_profile_publisher,
    set_active_profiler,
    set_profile_publisher,
)
from flwr.common.retry_invoker import _make_simple_grpc_retry_invoker, _wrap_stub
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
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub
from flwr.server.grid.grpc_grid import GrpcGrid
from flwr.server.run_serverapp import run as run_
from flwr.supercore.app_utils import start_parent_process_monitor
from flwr.supercore.heartbeat import HeartbeatSender, make_app_heartbeat_fn_grpc
from flwr.supercore.superexec.plugin import ServerAppExecPlugin
from flwr.supercore.superexec.run_superexec import run_with_deprecation_warning


def _resolve_profile_enabled(
    run_config: dict[str, Any], override_config: dict[str, Any]
) -> bool:
    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    for key in ("profile.enabled", "profiling.enabled"):
        if key in override_config:
            return _to_bool(override_config[key])
    for key in ("profile.enabled", "profiling.enabled"):
        if key in run_config:
            return _to_bool(run_config[key])
    return False


def flwr_serverapp() -> None:
    """Run process-isolated Flower ServerApp."""
    # Capture stdout/stderr
    log_queue: Queue[str | None] = Queue()
    mirror_output_to_queue(log_queue)

    args = _parse_args_run_flwr_serverapp().parse_args()

    if not args.insecure:
        flwr_exit(
            ExitCode.COMMON_TLS_NOT_SUPPORTED,
            "`flwr-serverapp` does not support TLS yet.",
        )

    # Disallow long-running `flwr-serverapp` processes
    if args.token is None:
        run_with_deprecation_warning(
            cmd="flwr-serverapp",
            plugin_type=ExecPluginType.SERVER_APP,
            plugin_class=ServerAppExecPlugin,
            stub_class=ServerAppIoStub,
            appio_api_address=args.serverappio_api_address,
            flwr_dir=args.flwr_dir,
            parent_pid=args.parent_pid,
            warn_run_once=args.run_once,
        )
        return

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
        flwr_dir=args.flwr_dir,
        certificates=None,
        parent_pid=args.parent_pid,
    )

    # Restore stdout/stderr
    restore_output()


def run_serverapp(  # pylint: disable=R0913, R0914, R0915, R0917, W0212
    serverappio_api_address: str,
    log_queue: Queue[str | None],
    token: str,
    flwr_dir: str | None = None,
    certificates: bytes | None = None,
    parent_pid: int | None = None,
) -> None:
    """Run Flower ServerApp process."""
    # Monitor the main process in case of SIGKILL
    if parent_pid is not None:
        start_parent_process_monitor(parent_pid)

    # Initialize variables for exit handler
    flwr_dir_ = get_flwr_dir(flwr_dir)
    log_uploader = None
    hash_run_id = None
    run = None
    run_status = None
    heartbeat_sender = None
    heartbeat_channel = None
    grid = None
    context = None
    profile_recorder: ProfileRecorder | None = None
    exit_code = ExitCode.SUCCESS

    def on_exit() -> None:
        # Stop heartbeat sender
        if heartbeat_sender and heartbeat_sender.is_running:
            heartbeat_sender.stop()
        if heartbeat_channel:
            heartbeat_channel.close()

        # Stop log uploader for this run and upload final logs
        if log_uploader:
            stop_log_uploader(log_queue, log_uploader)

        # Update run status
        if run and run_status and grid:
            run_status_proto = run_status_to_proto(run_status)
            grid._stub.UpdateRunStatus(
                UpdateRunStatusRequest(run_id=run.run_id, run_status=run_status_proto)
            )

        # Close the Grpc connection
        if grid:
            grid.close()

    # Register signal handlers for graceful shutdown
    register_signal_handlers(
        event_type=EventType.FLWR_SERVERAPP_RUN_LEAVE,
        exit_message="Run stopped by user.",
        exit_handlers=[on_exit],
    )

    try:
        # Initialize the GrpcGrid (use default pull interval until run_config loaded)
        grid = GrpcGrid(
            serverappio_service_address=serverappio_api_address,
            root_certificates=certificates,
            pull_interval=3.0,
        )

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

        grid.set_run(run.run_id)

        # Start log uploader for this run
        log_uploader = start_log_uploader(
            log_queue=log_queue,
            node_id=0,
            run_id=run.run_id,
            stub=grid._stub,
        )

        log(DEBUG, "[flwr-serverapp] Start FAB installation.")
        install_from_fab(fab.content, flwr_dir=flwr_dir_, skip_prompt=True)

        fab_id, fab_version = get_fab_metadata(fab.content)

        app_path = str(get_project_dir(fab_id, fab_version, fab.hash_str, flwr_dir_))
        config = get_project_config(app_path)

        # Obtain server app reference and the run config
        server_app_attr = config["tool"]["flwr"]["app"]["components"]["serverapp"]
        server_app_run_config = get_fused_config_from_dir(
            Path(app_path), run.override_config
        )

        # Update run_config in context
        context.run_config = server_app_run_config
        grid.pull_interval = float(server_app_run_config.get("grid.pull-interval", 3.0))

        # Ensure profile.enabled reflects overrides (even if not in defaults)
        if (
            "profile.enabled" in run.override_config
            or "profiling.enabled" in run.override_config
        ):
            context.run_config["profile.enabled"] = _resolve_profile_enabled(
                context.run_config, run.override_config
            )

        # Initialize profiler if enabled
        profile_enabled = _resolve_profile_enabled(
            context.run_config, run.override_config
        )
        log(INFO, "[flwr-serverapp] profile.enabled resolved to %s", profile_enabled)
        log(INFO, "[flwr-serverapp] module path: %s", __file__)
        log(INFO, "[flwr-serverapp] python: %s", os.getenv("PYTHON_EXECUTABLE", ""))
        if profile_enabled:
            profile_recorder = ProfileRecorder(run_id=run.run_id)
            set_active_profiler(profile_recorder)
            last_summary: bytes | None = None

            def _publish(summary: dict[str, object]) -> None:
                nonlocal last_summary
                summary_json = json.dumps(summary).encode()
                if summary_json == last_summary:
                    return
                last_summary = summary_json
                context.state["profile_summary"] = ConfigRecord({"json": summary_json})
                context.state["profile_live"] = ConfigRecord(
                    {"enabled": True, "final": False}
                )
                out_req = PushAppOutputsRequest(
                    token=token, run_id=run.run_id, context=context_to_proto(context)
                )
                _ = grid._stub.PushAppOutputs(out_req)

            set_profile_publisher(_publish)

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

        # Set up heartbeat sender
        heartbeat_channel = create_channel(
            server_address=serverappio_api_address,
            insecure=(certificates is None),
            root_certificates=certificates,
        )
        heartbeat_stub = ServerAppIoStub(heartbeat_channel)
        _wrap_stub(heartbeat_stub, _make_simple_grpc_retry_invoker())
        heartbeat_sender = HeartbeatSender(
            make_app_heartbeat_fn_grpc(heartbeat_stub, token)
        )
        heartbeat_sender.start()

        # Load and run the ServerApp with the Grid
        updated_context = run_(
            grid=grid,
            server_app_dir=app_path,
            server_app_attr=server_app_attr,
            context=context,
        )

        # Send resulting context
        if profile_recorder is not None:
            summary_json = json.dumps(profile_recorder.summarize()).encode()
            updated_context.state["profile_summary"] = ConfigRecord(
                {"json": summary_json}
            )
            updated_context.state["profile_live"] = ConfigRecord(
                {"enabled": True, "final": True}
            )
            clear_profile_publisher()
            clear_active_profiler()
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
