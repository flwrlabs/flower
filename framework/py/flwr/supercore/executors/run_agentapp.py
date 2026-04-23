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


from pathlib import Path
from queue import Queue

from flwr.common import EventType
from flwr.common.constant import RUNTIME_DEPENDENCY_INSTALL
from flwr.common.exit import ExitCode, flwr_exit, register_signal_handlers
from flwr.common.logger import stop_log_uploader
from flwr.supercore.app_utils import start_parent_process_monitor
from flwr.supercore.superexec.dependency_installer import (
    cleanup_app_runtime_environment,
)


def run_agentapp(  # pylint: disable=R0913, R0917
    serverappio_api_address: str,
    log_queue: Queue[str | None],
    token: str,
    certificates: bytes | None = None,
    parent_pid: int | None = None,
    runtime_dependency_install: bool = RUNTIME_DEPENDENCY_INSTALL,
) -> None:
    """Run Flower AgentApp process.

    This runtime is intentionally a stub until AgentApp execution support is added.
    """
    # Monitor the main process in case of SIGKILL
    if parent_pid is not None:
        start_parent_process_monitor(parent_pid)

    log_uploader = None
    runtime_env_dir: Path | None = None

    def on_exit() -> None:
        if log_uploader:
            stop_log_uploader(log_queue, log_uploader)
        cleanup_app_runtime_environment(runtime_env_dir)

    register_signal_handlers(
        event_type=EventType.FLWR_AGENTAPP_RUN_LEAVE,
        exit_message="Run stopped by user.",
        exit_handlers=[on_exit],
    )

    _ = (
        serverappio_api_address,
        log_queue,
        token,
        certificates,
        parent_pid,
        runtime_dependency_install,
    )
    flwr_exit(
        ExitCode.SERVERAPP_EXCEPTION,
        "`flwr-agentapp` is not implemented yet.",
        event_type=EventType.FLWR_AGENTAPP_RUN_LEAVE,
        event_details={"success": False},
    )
