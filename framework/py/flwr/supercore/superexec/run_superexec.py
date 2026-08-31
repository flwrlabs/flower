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


import math
import os
import time
from logging import ERROR, WARNING
from typing import Any

from flwr.common.constant import RUNTIME_DEPENDENCY_INSTALL
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    ClaimTaskRequest,
    PullPendingTasksRequest,
)
from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.supercore import log
from flwr.supercore.app_utils import start_parent_process_monitor
from flwr.supercore.constant import ExecutorType
from flwr.supercore.exit import ExitCode, flwr_exit, register_signal_handlers
from flwr.supercore.grpc_health import run_health_server_grpc_no_tls
from flwr.supercore.interceptors import (
    RuntimeVersionHttpInterceptor,
    SuperExecAuthHttpInterceptor,
)
from flwr.supercore.interceptors.superexec_auth_interceptor import (
    RUNTIME_SUPEREXEC_METHODS,
)
from flwr.supercore.protobuf.client import ProtobufClientInterceptor
from flwr.supercore.retry import make_simple_http_retry_invoker
from flwr.supercore.runtime import RuntimeHttpClient
from flwr.supercore.telemetry import EventType
from flwr.supercore.tls import validate_and_resolve_root_certificates

from .executor import LaunchResult, LaunchResultStatus, get_executor
from .executor.config import ExecutorConfig
from .plugin import ExecPlugin
from .plugin.base_ephemeral_exec_plugin import BaseEphemeralExecPlugin

_TASK_POLL_INTERVAL_ENV = "FLWR_SUPEREXEC_TASK_POLL_INTERVAL"
_DEFAULT_TASK_POLL_INTERVAL_SECONDS = 0.1


def _get_task_poll_interval() -> float:
    """Read and validate the SuperExec task polling interval."""
    raw_interval = os.getenv(_TASK_POLL_INTERVAL_ENV)
    if raw_interval is None:
        return _DEFAULT_TASK_POLL_INTERVAL_SECONDS

    try:
        interval = float(raw_interval)
    except ValueError as err:
        raise ValueError(
            f"Environment variable {_TASK_POLL_INTERVAL_ENV} must be a positive "
            f"finite number of seconds, got {raw_interval!r}."
        ) from err

    if not math.isfinite(interval) or interval <= 0:
        raise ValueError(
            f"Environment variable {_TASK_POLL_INTERVAL_ENV} must be a positive "
            f"finite number of seconds, got {raw_interval!r}."
        )

    return interval


def _handle_launch_result(result: LaunchResult | None, task: Task) -> None:
    """Handle the immediate outcome of a TaskExecutor launch attempt."""
    # Temporary: ephemeral plugins may not return a LaunchResult.
    # Remove this once ephemeral plugins are removed.
    if result is None:
        return

    if result.status == LaunchResultStatus.ACCEPTED:
        return

    message = result.message or "Not provided by executor."
    if result.status == LaunchResultStatus.CAPACITY_REJECTED:
        log(
            WARNING,
            "Executor rejected launch for task_id %d due to capacity. Reason: %s "
            "Existing task expiry handling will apply.",
            task.task_id,
            message,
        )
        return

    if result.status == LaunchResultStatus.FAILED:
        log(
            ERROR,
            "Executor failed to launch task_id %d. Reason: %s "
            "Existing task expiry handling will apply.",
            task.task_id,
            message,
        )
        return

    if result.status == LaunchResultStatus.UNKNOWN:
        log(
            WARNING,
            "Executor launch outcome is unknown for task_id %d. Reason: %s "
            "Existing task expiry handling will apply.",
            task.task_id,
            message,
        )
        return

    raise RuntimeError(
        f"Executor returned unrecognized launch result '{result.status}' "
        f"for task_id {task.task_id}. Reason: {message}"
    )


def run_superexec(  # pylint: disable=R0912,R0913,R0914,R0915,R0917
    plugin_class: type[ExecPlugin],
    client_class: type[RuntimeHttpClient],
    runtime_api_address: str,
    insecure: bool,
    root_certificates_path: str | None = None,
    superexec_auth_secret: bytes | None = None,
    plugin_config: dict[str, Any] | None = None,
    parent_pid: int | None = None,
    health_server_address: str | None = None,
    runtime_dependency_install: bool = RUNTIME_DEPENDENCY_INSTALL,
    executor_type: ExecutorType = ExecutorType.SUBPROCESS,
    executor_config: ExecutorConfig | None = None,
) -> None:
    """Run Flower SuperExec.

    Parameters
    ----------
    plugin_class : type[ExecPlugin]
        The class of the SuperExec plugin to use.
    client_class : type[RuntimeHttpClient]
        The HTTP client class for the Runtime API.
    runtime_api_address : str
        The address of the Runtime API.
    insecure : bool
        Whether to connect to the Runtime API without TLS.
    root_certificates_path : Optional[str] (default: None)
        The path to the PEM-encoded root certificate file used for secure TLS
        connections.
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
    runtime_dependency_install : bool (default: False)
        Whether runtime dependency installation is allowed.
    executor_type : ExecutorType (default: ExecutorType.SUBPROCESS)
        The executor to use for non-ephemeral app processes.
    executor_config : Optional[ExecutorConfig] (default: None)
        Parsed executor configuration.
    """
    task_poll_interval = _get_task_poll_interval()

    try:
        executor = get_executor(executor_type, executor_config=executor_config)
    except ValueError as err:
        flwr_exit(ExitCode.SUPEREXEC_INVALID_EXECUTOR_CONFIG, str(err))

    interceptors: list[ProtobufClientInterceptor] = [
        RuntimeVersionHttpInterceptor(component_name="SuperExec")
    ]
    auth_interceptor: SuperExecAuthHttpInterceptor | None = None
    if superexec_auth_secret:
        auth_interceptor = SuperExecAuthHttpInterceptor(
            master_secret=superexec_auth_secret,
            protected_methods=RUNTIME_SUPEREXEC_METHODS,
        )
        interceptors.append(auth_interceptor)

    # Start monitoring the parent process if a PID is provided
    if parent_pid is not None:
        start_parent_process_monitor(parent_pid)

    # Launch gRPC health server
    grpc_servers = []
    if health_server_address is not None:
        health_server = run_health_server_grpc_no_tls(health_server_address)
        grpc_servers.append(health_server)

    client = client_class.from_server_address(
        server_address=runtime_api_address,
        insecure=insecure,
        root_certificates=validate_and_resolve_root_certificates(
            root_certificates_path, insecure
        ),
        interceptors=interceptors,
        retry_invoker=make_simple_http_retry_invoker(),
    )

    # Register exit handlers to close the Runtime API client on exit
    register_signal_handlers(
        event_type=EventType.RUN_SUPEREXEC_LEAVE,
        exit_message="SuperExec terminated gracefully.",
        grpc_servers=grpc_servers,
        exit_handlers=[client.close],
    )

    # Create the SuperExec plugin instance
    plugin = plugin_class(
        runtime_api_address=runtime_api_address,
        insecure=insecure,
        root_certificates_path=root_certificates_path,
        runtime_dependency_install=runtime_dependency_install,
        executor=executor,
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
            # Fetch pending tasks
            tasks_res = client.PullPendingTasks(request=PullPendingTasksRequest())

            # Select a task to execute using the plugin's selection logic
            task = None
            if tasks_res.tasks:
                task = plugin.select_task(tasks_res.tasks)

            # If a task was selected, claim it
            if task is not None:
                executor.wait_for_capacity()

                claim_req = ClaimTaskRequest(task_id=task.task_id)
                claim_res = client.ClaimTask(claim_req)

                # Launch the app if a token was granted; do nothing if not
                if claim_res.token:

                    # Destroy the auth secret before launching the app
                    # for ephemeral plugins
                    if isinstance(plugin, BaseEphemeralExecPlugin):

                        def cleanup_auth_secret() -> None:
                            nonlocal superexec_auth_secret
                            if superexec_auth_secret is not None:
                                superexec_auth_secret = None
                            if auth_interceptor is not None:
                                # pylint: disable-next=protected-access
                                auth_interceptor._auth_secret = b"\x00" * 32

                        plugin.cleanup_before_launch = cleanup_auth_secret

                    launch_result = plugin.launch_task(token=claim_res.token, task=task)
                    _handle_launch_result(launch_result, task)

            # Sleep for a while before checking again
            time.sleep(task_poll_interval)
    finally:
        client.close()
