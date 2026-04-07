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
"""Flower Simulation process."""


import argparse
from dataclasses import replace
from logging import DEBUG, ERROR, INFO
from queue import Queue

from flwr.app import RecordDict
from flwr.cli.config_utils import get_fab_metadata
from flwr.cli.install import install_from_fab
from flwr.cli.utils import get_sha256_hash
from flwr.common import EventType, event
from flwr.common.args import add_args_flwr_app_common
from flwr.common.config import (
    get_fused_config_from_dir,
    get_project_config,
    get_project_dir,
)
from flwr.common.constant import (
    RUNTIME_DEPENDENCY_INSTALL,
    SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS,
    ExecPluginType,
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
from flwr.common.typing import RunStatus
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    PullAppInputsRequest,
    PullAppInputsResponse,
    PushAppOutputsRequest,
)
from flwr.proto.federation_config_pb2 import SimulationConfig  # pylint: disable=E0611
from flwr.proto.run_pb2 import UpdateRunStatusRequest  # pylint: disable=E0611
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub
from flwr.server.superlink.fleet.vce.backend.backend import BackendConfig
from flwr.simulation.run_simulation import _run_simulation
from flwr.simulation.simulationio_connection import SimulationIoConnection
from flwr.supercore.app_utils import start_parent_process_monitor
from flwr.supercore.constant import NOOP_FEDERATION
from flwr.supercore.heartbeat import HeartbeatSender, make_app_heartbeat_fn_grpc
from flwr.supercore.superexec.dependency_installer import (
    cleanup_app_runtime_environment,
    install_app_dependencies,
)
from flwr.supercore.superexec.plugin import ServerAppExecPlugin
from flwr.supercore.superexec.run_superexec import run_with_deprecation_warning


def _run_simulation_settings(
    sim_cfg: SimulationConfig,
) -> tuple[int, str, BackendConfig, bool, bool]:
    """Extract simulation runtime settings from a run."""
    if sim_cfg is None or not sim_cfg.HasField("num_supernodes"):
        raise ValueError(
            "Simulation run expects `run.federation_config.num_supernodes` to be set."
        )

    backend_name = sim_cfg.backend if sim_cfg.HasField("backend") else "ray"
    backend_config: BackendConfig = {"client_resources": {}, "init_args": {}}

    if sim_cfg.HasField("client_resources_num_cpus"):
        backend_config["client_resources"][
            "num_cpus"
        ] = sim_cfg.client_resources_num_cpus
    if sim_cfg.HasField("client_resources_num_gpus"):
        backend_config["client_resources"][
            "num_gpus"
        ] = sim_cfg.client_resources_num_gpus
    if sim_cfg.HasField("init_args_num_cpus"):
        backend_config["init_args"]["num_cpus"] = sim_cfg.init_args_num_cpus
    if sim_cfg.HasField("init_args_num_gpus"):
        backend_config["init_args"]["num_gpus"] = sim_cfg.init_args_num_gpus
    if sim_cfg.HasField("init_args_logging_level"):
        backend_config["init_args"]["logging_level"] = sim_cfg.init_args_logging_level
    if sim_cfg.HasField("init_args_log_to_driver"):
        backend_config["init_args"]["log_to_driver"] = sim_cfg.init_args_log_to_driver

    verbose = sim_cfg.verbose if sim_cfg.HasField("verbose") else False
    return sim_cfg.num_supernodes, backend_name, backend_config, verbose, False


def flwr_simulation() -> None:
    """Run process-isolated Flower Simulation."""
    # Capture stdout/stderr
    log_queue: Queue[str | None] = Queue()
    mirror_output_to_queue(log_queue)

    args = _parse_args_run_flwr_simulation().parse_args()

    if not args.insecure:
        flwr_exit(
            ExitCode.COMMON_TLS_NOT_SUPPORTED,
            "`flwr-simulation` does not support TLS yet.",
        )

    # Disallow long-running `flwr-simulation` processes
    if args.token is None:
        run_with_deprecation_warning(
            cmd="flwr-simulation",
            plugin_type=ExecPluginType.SERVER_APP,
            plugin_class=ServerAppExecPlugin,
            stub_class=ServerAppIoStub,
            appio_api_address=args.serverappio_api_address,
            parent_pid=args.parent_pid,
            warn_run_once=args.run_once,
            runtime_dependency_install=args.runtime_dependency_install,
        )
        return

    log(INFO, "Starting Flower Simulation")
    log(
        DEBUG,
        "Starting isolated `Simulation` connected to SuperLink ServerAppIo API at %s",
        args.serverappio_api_address,
    )
    run_simulation_process(
        serverappio_api_address=args.serverappio_api_address,
        log_queue=log_queue,
        token=args.token,
        certificates=None,
        parent_pid=args.parent_pid,
        runtime_dependency_install=args.runtime_dependency_install,
    )

    # Restore stdout/stderr
    restore_output()


def run_simulation_process(  # pylint: disable=R0913, R0914, R0915, R0917, W0212
    serverappio_api_address: str,
    log_queue: Queue[str | None],
    token: str,
    certificates: bytes | None = None,
    parent_pid: int | None = None,
    runtime_dependency_install: bool = RUNTIME_DEPENDENCY_INSTALL,
) -> None:
    """Run Flower Simulation process."""
    # Start monitoring the parent process if a PID is provided
    if parent_pid is not None:
        start_parent_process_monitor(parent_pid)

    conn = SimulationIoConnection(
        serverappio_api_address=serverappio_api_address,
        root_certificates=certificates,
        token=token,
    )

    # Initialize variables for finally block
    log_uploader = None
    run_id_hash = None
    heartbeat_sender = None
    run = None
    run_status = None
    run_id_hash = None
    runtime_env_dir = None
    exit_code = ExitCode.SUCCESS

    def on_exit() -> None:
        # Stop heartbeat sender
        if heartbeat_sender and heartbeat_sender.is_running:
            heartbeat_sender.stop()

        # Stop log uploader for this run and upload final logs
        if log_uploader:
            stop_log_uploader(log_queue, log_uploader)

        # Update run status
        if run and run_status:
            run_status_proto = run_status_to_proto(run_status)
            conn._stub.UpdateRunStatus(
                UpdateRunStatusRequest(run_id=run.run_id, run_status=run_status_proto)
            )

        cleanup_app_runtime_environment(runtime_env_dir)

    register_signal_handlers(
        event_type=EventType.FLWR_SIMULATION_RUN_LEAVE,
        exit_message="Run stopped by user.",
        exit_handlers=[on_exit],
    )

    try:
        # Pull SimulationInputs from LinkState
        req = PullAppInputsRequest(token=token)
        res: PullAppInputsResponse = conn._stub.PullAppInputs(req)
        context = context_from_proto(res.context)
        run = run_from_proto(res.run)
        fab = fab_from_proto(res.fab)

        # Start log uploader for this run
        log_uploader = start_log_uploader(
            log_queue=log_queue,
            node_id=context.node_id,
            run_id=run.run_id,
            stub=conn._stub,
        )

        log(DEBUG, "Simulation process starts FAB installation.")
        install_from_fab(fab.content, skip_prompt=True)

        fab_id, fab_version = get_fab_metadata(fab.content)

        app_path = get_project_dir(fab_id, fab_version, fab.hash_str)

        if runtime_dependency_install:
            log(DEBUG, "Simulation process starts app dependency installation.")
            runtime_env_dir = install_app_dependencies(
                app_path,
                launch_id=token,
                run_id=run.run_id,
                index_context={
                    "component": "simulation",
                    "project_dir": str(app_path),
                    "run_id": run.run_id,
                    "launch_id": token,
                    "fab_id": run.fab_id,
                    "fab_version": run.fab_version,
                    "fab_hash": fab.hash_str,
                },
            )
        else:
            log(DEBUG, "Simulation runtime dependency installation is disabled.")

        config = get_project_config(app_path)

        # Get ClientApp and SeverApp components
        app_components = config["tool"]["flwr"]["app"]["components"]
        client_app_attr = app_components["clientapp"]
        server_app_attr = app_components["serverapp"]
        fused_config = get_fused_config_from_dir(app_path, run.override_config)

        # Update run_config in context
        context.run_config = fused_config

        log(
            DEBUG,
            "Flower will load ServerApp `%s` in %s",
            server_app_attr,
            app_path,
        )
        log(
            DEBUG,
            "Flower will load ClientApp `%s` in %s",
            client_app_attr,
            app_path,
        )

        (
            num_supernodes,
            backend_name,
            backend_config,
            verbose,
            enable_tf_gpu_growth,
        ) = _run_simulation_settings(res.federation_config)

        run_id_hash = get_sha256_hash(run.run_id)
        event(
            EventType.FLWR_SIMULATION_RUN_ENTER,
            event_details={
                "backend": backend_name,
                "num-supernodes": num_supernodes,
                "run-id-hash": run_id_hash,
            },
        )

        # Set up heartbeat sender
        heartbeat_sender = HeartbeatSender(
            make_app_heartbeat_fn_grpc(conn._stub, token)
        )
        heartbeat_sender.start()

        # Launch the simulation
        updated_context = _run_simulation(
            server_app_attr=server_app_attr,
            client_app_attr=client_app_attr,
            num_supernodes=num_supernodes,
            backend_name=backend_name,
            backend_config=backend_config,
            app_dir=str(app_path),
            run=replace(run, federation=NOOP_FEDERATION),
            enable_tf_gpu_growth=enable_tf_gpu_growth,
            verbose_logging=verbose,
            server_app_context=context,
            is_app=True,
            exit_event=EventType.FLWR_SIMULATION_RUN_LEAVE,
        )

        # Send resulting context
        # Temporarily disable pushing resulting context to SuperLink
        updated_context.state = RecordDict()
        context_proto = context_to_proto(updated_context)
        out_req = PushAppOutputsRequest(
            token=token, run_id=run.run_id, context=context_proto
        )
        _ = conn._stub.PushAppOutputs(out_req)

        run_status = RunStatus(Status.FINISHED, SubStatus.COMPLETED, "")

    except Exception as ex:  # pylint: disable=broad-exception-caught
        exc_entity = "Simulation"
        log(ERROR, "%s raised an exception", exc_entity, exc_info=ex)
        run_status = RunStatus(Status.FINISHED, SubStatus.FAILED, str(ex))

        # General exit code
        exit_code = ExitCode.SIMULATION_EXCEPTION

    flwr_exit(
        code=exit_code,
        event_type=EventType.FLWR_SIMULATION_RUN_LEAVE,
        event_details={
            "run-id-hash": run_id_hash,
            "success": exit_code == ExitCode.SUCCESS,
        },
    )


def _parse_args_run_flwr_simulation() -> argparse.ArgumentParser:
    """Parse flwr-simulation command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a Flower Simulation",
    )
    parser.add_argument(
        "--serverappio-api-address",
        "--simulationio-api-address",
        dest="serverappio_api_address",
        default=SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS,
        type=str,
        help="Address of SuperLink's ServerAppIo API (IPv4, IPv6, or a domain name). "
        "`--simulationio-api-address` is accepted as a deprecated alias. "
        f"By default, it is set to {SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS}.",
    )
    add_args_flwr_app_common(parser=parser)
    return parser
