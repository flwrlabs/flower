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
"""Flower SuperExec."""


import time
from logging import WARN
from typing import Any

from flwr.common.exit import ExitCode, flwr_exit, register_signal_handlers
from flwr.common.grpc import create_channel, on_channel_state_change
from flwr.common.logger import log
from flwr.common.retry_invoker import make_simple_grpc_retry_invoker, wrap_stub
from flwr.common.serde import run_from_proto
from flwr.common.telemetry import EventType
from flwr.common.typing import Run
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    ListAppsToLaunchRequest,
    RequestTokenRequest,
)
from flwr.proto.clientappio_pb2_grpc import ClientAppIoStub
from flwr.proto.run_pb2 import GetRunRequest  # pylint: disable=E0611
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub
from flwr.supercore.app_utils import start_parent_process_monitor
from flwr.supercore.grpc_health import run_health_server_grpc_no_tls
from flwr.supercore.interceptors import SuperExecAuthClientInterceptor
from flwr.supercore.interceptors.superexec_auth_interceptor import (
    SERVERAPPIO_SUPEREXEC_METHODS,
)

from .plugin import ExecPlugin


def run_superexec(  # pylint: disable=R0913,R0914,R0917
    plugin_class: type[ExecPlugin],
    stub_class: type[ClientAppIoStub] | type[ServerAppIoStub],
    appio_api_address: str,
    superexec_auth_secret: bytes | None = None,
    plugin_config: dict[str, Any] | None = None,
    parent_pid: int | None = None,
    health_server_address: str | None = None,
) -> None:
    """Run Flower SuperExec.

    Parameters
    ----------
    plugin_class : type[ExecPlugin]
        The class of the SuperExec plugin to use.
    stub_class : type[ClientAppIoStub] | type[ServerAppIoStub]
        The gRPC stub class for the AppIO API.
    appio_api_address : str
        The address of the AppIO API.
    superexec_auth_secret : Optional[bytes] (default: None)
        Secret used to derive an HMAC signing key for SuperExec auth.
    plugin_config : Optional[dict[str, Any]] (default: None)
        The configuration dictionary for the plugin. If `None`, the plugin will use
        its default configuration.
    parent_pid : Optional[int] (default: None)
        The PID of the parent process. If provided, the SuperExec will terminate
        when the parent process exits.
    health_server_address : Optional[str] (default: None)
        The address of the health server. If `None` is provided, the health server will
        NOT be started.
    """
    interceptors: list[SuperExecAuthClientInterceptor] | None = None
    if stub_class is ServerAppIoStub and superexec_auth_secret:
        interceptors = [
            SuperExecAuthClientInterceptor(
                master_secret=superexec_auth_secret,
                protected_methods=SERVERAPPIO_SUPEREXEC_METHODS,
            )
        ]

    # Start monitoring the parent process if a PID is provided
    if parent_pid is not None:
        start_parent_process_monitor(parent_pid)

    # Launch gRPC health server
    grpc_servers = []
    if health_server_address is not None:
        health_server = run_health_server_grpc_no_tls(health_server_address)
        grpc_servers.append(health_server)

    # Create the channel to the AppIO API
    # No TLS support for now, so insecure connection
    channel = create_channel(
        server_address=appio_api_address,
        insecure=True,
        root_certificates=None,
        interceptors=interceptors,
    )
    channel.subscribe(on_channel_state_change)

    # Register exit handlers to close the channel on exit
    register_signal_handlers(
        event_type=EventType.RUN_SUPEREXEC_LEAVE,
        exit_message="SuperExec terminated gracefully.",
        grpc_servers=grpc_servers,
        exit_handlers=[lambda: channel.close()],  # pylint: disable=W0108
    )

    # Create the gRPC stub for the AppIO API
    stub = stub_class(channel)
    wrap_stub(stub, make_simple_grpc_retry_invoker())

    def get_run(run_id: int) -> Run:
        _req = GetRunRequest(run_id=run_id)
        _res = stub.GetRun(_req)
        return run_from_proto(_res.run)

    # Create the SuperExec plugin instance
    plugin = plugin_class(
        appio_api_address=appio_api_address,
        get_run=get_run,
    )

    # Load plugin configuration from file if provided
    try:
        if plugin_config is not None:
            plugin.load_config(plugin_config)
    except (KeyError, ValueError) as e:
        flwr_exit(
            code=ExitCode.SUPEREXEC_INVALID_PLUGIN_CONFIG,
            message=f"Invalid plugin config: {e!r}",
        )

    # Start the main loop
    try:
        while True:
            # Fetch suitable run IDs
            ls_req = ListAppsToLaunchRequest()
            ls_res = stub.ListAppsToLaunch(ls_req)

            # Allow the plugin to select a run ID
            run_id = None
            if ls_res.run_ids:
                run_id = plugin.select_run_id(candidate_run_ids=ls_res.run_ids)

            # Apply for a token if a run ID was selected
            if run_id is not None:
                tk_req = RequestTokenRequest(run_id=run_id)
                tk_res = stub.RequestToken(tk_req)

                # Launch the app if a token was granted; do nothing if not
                if tk_res.token:
                    plugin.launch_app(token=tk_res.token, run_id=run_id)

            # Sleep for a while before checking again
            time.sleep(1)
    finally:
        channel.close()


def run_with_deprecation_warning(  # pylint: disable=R0913, R0917
    cmd: str,
    plugin_type: str,
    plugin_class: type[ExecPlugin],
    stub_class: type[ClientAppIoStub] | type[ServerAppIoStub],
    appio_api_address: str,
    parent_pid: int | None,
    warn_run_once: bool,
    superexec_auth_secret: bytes | None = None,
) -> None:
    """Log a deprecation warning and run the equivalent `flower-superexec` command.

    Used for legacy long-running `flwr-*` commands (i.e., without `--token`) that will
    be removed in favor of `flower-superexec`.
    """
    log(
        WARN,
        "Directly executing `%s` is DEPRECATED and will be prohibited "
        "in a future release. Please use `flower-superexec` instead.",
        cmd,
    )
    log(WARN, "For now, the following command is being run automatically:")
    new_cmd = f"flower-superexec --insecure --plugin-type {plugin_type} "
    new_cmd += f"--appio-api-address {appio_api_address} "
    if parent_pid is not None:
        new_cmd += f"--parent-pid {parent_pid}"
    log(WARN, new_cmd)

    # Warn about unsupported `--run-once` flag
    if warn_run_once:
        log(WARN, "`flower-superexec` does not support the `--run-once` flag.")

    run_superexec(
        plugin_class=plugin_class,
        stub_class=stub_class,
        appio_api_address=appio_api_address,
        superexec_auth_secret=superexec_auth_secret,
        parent_pid=parent_pid,
    )
