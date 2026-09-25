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


import os
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from logging import DEBUG, ERROR
from pathlib import Path
from queue import Queue
from typing import Any, cast

from flwr.agentapp import AgentApp, LoadAgentAppError
from flwr.app import Context, Message
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
from flwr.common.serde import (
    context_from_proto,
    context_to_proto,
    fab_from_proto,
    fab_to_proto,
    run_from_proto,
    user_config_to_proto,
)
from flwr.proto.control_pb2 import StartRunRequest  # pylint: disable=E0611
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    PullTaskInputRequest,
    PullTaskInputResponse,
    PushTaskOutputRequest,
)
from flwr.supercore import log
from flwr.supercore.app_utils import start_parent_process_monitor
from flwr.supercore.constant import (
    AGENT_MESSAGE_CONTENT_RECORD_KEY,
    AGENT_MESSAGE_TEXT_KEY,
    SYSTEM_MESSAGE_TYPE,
)
from flwr.supercore.exit import ExitCode, flwr_exit, register_signal_handlers
from flwr.supercore.exit.signal_handler import SIGNAL_TO_EXIT_CODE
from flwr.supercore.heartbeat import HeartbeatSender, make_task_heartbeat_fn_http
from flwr.supercore.logger import flush_logs, start_log_uploader, stop_log_uploader
from flwr.supercore.object_ref import load_app
from flwr.supercore.superexec.dependency_installer import (
    RuntimeDependencyInstallationError,
    cleanup_app_runtime_environment,
    install_app_dependencies,
)
from flwr.supercore.task_identity import TaskIdentity
from flwr.supercore.telemetry import EventType, event
from flwr.supercore.tls import validate_and_resolve_root_certificates
from flwr.supercore.typing import JSONObject
from flwr.supercore.utils import strict_json_dumps
from flwr.superlink.grid import HttpGrid

from .grid import RuntimeAgentGrid
from .session import (
    AgentRuntime,
    RuntimeAgentConnectors,
    RuntimeAgentEvents,
    RuntimeAgentSession,
)

_RUNTIME_API_KEY_ENV = "FLWR_RUNTIME_API_KEY"
_RUNTIME_BASE_URL_ENV = "FLWR_RUNTIME_BASE_URL"
_SSL_CERT_FILE_ENV = "SSL_CERT_FILE"


def message_to_prompt(message: Message) -> str:
    """Serialize a Grid message into a JSON prompt string."""
    prompt: JSONObject = {
        "message_id": message.metadata.message_id,
        "src_node_id": str(message.metadata.src_node_id),
        "payload": cast(
            str,
            message.content[AGENT_MESSAGE_CONTENT_RECORD_KEY][AGENT_MESSAGE_TEXT_KEY],
        ),
    }
    # Return the payload string directly if the message is a system message
    if message.metadata.message_type == SYSTEM_MESSAGE_TYPE:
        return cast(str, prompt["payload"])
    # Otherwise, return the full prompt as a compact JSON string
    return strict_json_dumps(prompt, compact=True)


def pull_prompt(grid: HttpGrid) -> str:
    """Pull and serialize the initial AgentApp instruction."""
    instructions = list(grid.pull_messages([]))
    if len(instructions) != 1:
        raise RuntimeError("Expected exactly one initial AgentApp instruction.")
    return message_to_prompt(instructions[0])


class _AgentAppTaskLifecycle:  # pylint: disable=too-many-instance-attributes,protected-access
    """Own task-scoped AgentApp state and exactly-once finalization."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        runtime_api_address: str,
        log_queue: Queue[str | None],
        token: str,
        insecure: bool,
        certificates: bytes | None,
        certificates_path: str | None,
        runtime_dependency_install: bool,
    ) -> None:
        self._runtime_api_address = runtime_api_address
        self._log_queue = log_queue
        self._token = token
        self._insecure = insecure
        self._certificates = certificates
        self._certificates_path = certificates_path
        self._runtime_dependency_install = runtime_dependency_install
        self._grid = HttpGrid(
            runtime_api_address=self._runtime_api_address,
            insecure=self._insecure,
            root_certificates=self._certificates,
            token=self._token,
        )
        self._log_uploader: threading.Thread | None = None
        self._hash_run_id: str | None = None
        self._outcome = (SubStatus.FAILED, "Task failed with unknown error.")
        self._heartbeat_sender: HeartbeatSender | None = None
        self._context: Context | None = None
        self._runtime_env_dir: Path | None = None
        self._agent_events: RuntimeAgentEvents | None = None
        self._finalized = False
        self._lock = threading.RLock()

    def run(self) -> int:  # pylint: disable=too-many-locals,too-many-statements
        """Execute the AgentApp task and return its Flower exit code."""
        exit_code = ExitCode.SUCCESS
        try:
            grid = self._grid

            self._heartbeat_sender = HeartbeatSender(
                make_task_heartbeat_fn_http(grid._runtime_client)
            )
            self._heartbeat_sender.start()

            log(DEBUG, "[flwr-agentapp] Pull task input")
            res: PullTaskInputResponse = grid._runtime_client.PullTaskInput(
                PullTaskInputRequest()
            )

            self._context = context_from_proto(res.context)
            run = run_from_proto(res.run)
            fab = fab_from_proto(res.fab)
            task_id = res.task_id
            TaskIdentity.task_id = task_id
            TaskIdentity.run_id = run.run_id
            TaskIdentity.node_id = self._context.node_id

            self._hash_run_id = get_sha256_hash(run.run_id)

            grid.set_run(run)

            self._log_uploader = start_log_uploader(
                log_queue=self._log_queue,
                node_id=0,
                run_id=run.run_id,
                client=grid._runtime_client,
            )

            # Initialize the AgentApp session
            prompt = pull_prompt(grid)
            self._agent_events = RuntimeAgentEvents(grid._runtime_client)
            self._agent_events.emit(
                {"type": "message", "role": "user", "content": prompt}
            )
            agent_runtime = AgentRuntime(
                stub=grid._runtime_client,
                run_id=self._context.run_id,
                task_id=task_id,
                start_run_request=StartRunRequest(
                    fab=fab_to_proto(fab),
                    override_config=user_config_to_proto(run.override_config),
                    override_federation_config=res.federation_config,
                    federation=run.federation_id,
                    series_id=run.series_id,
                ),
                events=self._agent_events,
            )
            agent = RuntimeAgentSession(
                prompt=prompt,
                connectors=RuntimeAgentConnectors(agent_runtime),
                events=self._agent_events,
                grid=RuntimeAgentGrid(grid, self._agent_events, self._context.node_id),
            )

            log(DEBUG, "[flwr-agentapp] Start FAB installation.")
            install_from_fab(fab.content, skip_prompt=True)

            fab_id, fab_version = get_fab_metadata(fab.content)

            app_path = str(get_project_dir(fab_id, fab_version, fab.hash_str))

            if self._runtime_dependency_install:
                log(DEBUG, "[flwr-agentapp] Installing app dependencies.")
                self._runtime_env_dir = install_app_dependencies(
                    app_path,
                    launch_id=self._token,
                    run_id=run.run_id,
                    index_context={
                        "component": "agentapp",
                        "project_dir": app_path,
                        "run_id": run.run_id,
                        "launch_id": self._token,
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
            self._context.run_config = get_fused_config_from_dir(
                Path(app_path), run.override_config
            )

            log(
                DEBUG,
                "[flwr-agentapp] Will load AgentApp `%s` in %s",
                agent_app_attr,
                app_path,
            )

            event(
                EventType.FLWR_AGENTAPP_RUN_ENTER,
                event_details={"run-id-hash": self._hash_run_id},
            )

            _set_runtime_environment(
                self._runtime_api_address,
                self._token,
                self._insecure,
                self._certificates_path,
            )

            # Load and run the AgentApp
            agent_app = load_app(agent_app_attr, LoadAgentAppError, app_path)
            if not isinstance(agent_app, AgentApp):
                raise LoadAgentAppError(
                    f"Attribute '{agent_app_attr}' is not of type "
                    f"'{AgentApp.__name__}'.",
                ) from None
            agent_app(agent=agent, context=self._context)
            self._agent_events.close()

            # Set sub_status and details for successful completion
            with self._lock:
                self._outcome = (SubStatus.COMPLETED, "")

        except Exception as ex:  # pylint: disable=broad-exception-caught
            log(ERROR, "AgentApp raised an exception", exc_info=ex)

            with self._lock:
                self._outcome = (
                    SubStatus.FAILED,
                    f"AgentApp failed with exception: {str(ex)}",
                )

            exit_code = ExitCode.TASK_PROC_EXCEPTION
            if isinstance(ex, AppExitException):
                exit_code = ex.exit_code
            elif isinstance(ex, ImportError):
                exit_code = ExitCode.COMMON_APP_IMPORT_ERROR
            elif isinstance(ex, RuntimeDependencyInstallationError):
                exit_code = ExitCode.COMMON_RUNTIME_DEPENDENCY_INSTALLATION_ERROR

        return exit_code

    def finalize(self) -> None:  # pylint: disable=protected-access
        """Push final status and release task state exactly once."""
        with _ignore_graceful_signals(), self._lock:
            if self._finalized:
                return
            self._finalized = True

            log(DEBUG, "[flwr-agentapp] Will push AgentApp task output")
            self._grid._retry_invoker.max_tries = 1

            if self._agent_events is not None:
                try:
                    self._agent_events.close(1)
                except Exception as err:  # pylint: disable=broad-exception-caught
                    log(ERROR, "Failed to close AgentApp event publisher", exc_info=err)

            try:
                if self._log_uploader:
                    flush_logs(self._log_queue)
            except Exception as err:  # pylint: disable=broad-exception-caught
                log(ERROR, "Failed to flush AgentApp task logs", exc_info=err)

            sub_status, details = self._outcome
            pushoutput_req = PushTaskOutputRequest(
                context=(context_to_proto(self._context) if self._context else None),
                sub_status=sub_status,
                details=details,
            )
            try:
                self._grid._runtime_client.PushTaskOutput(pushoutput_req)
            except Exception as err:  # pylint: disable=broad-exception-caught
                log(ERROR, "Failed to push AgentApp task output", exc_info=err)

            try:
                if self._log_uploader:
                    stop_log_uploader(self._log_queue, self._log_uploader)
            except Exception as err:  # pylint: disable=broad-exception-caught
                log(ERROR, "Failed to stop AgentApp log uploader", exc_info=err)

            try:
                if self._heartbeat_sender and self._heartbeat_sender.is_running:
                    self._heartbeat_sender.stop()
            except Exception as err:  # pylint: disable=broad-exception-caught
                log(ERROR, "Failed to stop AgentApp task heartbeat", exc_info=err)

            try:
                self._grid.close()
            except Exception as err:  # pylint: disable=broad-exception-caught
                log(ERROR, "Failed to close AgentApp Runtime client", exc_info=err)

            try:
                cleanup_app_runtime_environment(self._runtime_env_dir)
            except Exception as err:  # pylint: disable=broad-exception-caught
                log(ERROR, "Failed to clean up AgentApp runtime", exc_info=err)

    def event_details(self, exit_code: int) -> JSONObject:
        """Return the AgentApp leave-event details."""
        return {
            "run-id-hash": self._hash_run_id,
            "success": exit_code == ExitCode.SUCCESS,
        }


@contextmanager
def _ignore_graceful_signals() -> Iterator[None]:
    """Ignore graceful signals process-wide while finalization is in progress."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    previous_handlers: dict[int, Any] = {}
    try:
        for sig in SIGNAL_TO_EXIT_CODE:
            previous_handlers[sig] = signal.signal(sig, signal.SIG_IGN)
        yield
    finally:
        for sig, previous_handler in previous_handlers.items():
            signal.signal(sig, previous_handler)


def _run_agentapp_task(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    runtime_api_address: str,
    log_queue: Queue[str | None],
    token: str,
    insecure: bool,
    certificates: bytes | None,
    certificates_path: str | None,
    runtime_dependency_install: bool,
) -> tuple[_AgentAppTaskLifecycle, int]:
    """Create and execute one task through the AgentApp lifecycle."""
    lifecycle = _AgentAppTaskLifecycle(
        runtime_api_address=runtime_api_address,
        log_queue=log_queue,
        token=token,
        insecure=insecure,
        certificates=certificates,
        certificates_path=certificates_path,
        runtime_dependency_install=runtime_dependency_install,
    )
    register_signal_handlers(
        event_type=EventType.FLWR_AGENTAPP_RUN_LEAVE,
        exit_message="Task stopped by user.",
        exit_handlers=[lifecycle.finalize],
    )
    return lifecycle, lifecycle.run()


def run_agentapp(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    runtime_api_address: str,
    log_queue: Queue[str | None],
    token: str,
    insecure: bool,
    certificates_path: str | None = None,
    parent_pid: int | None = None,
    runtime_dependency_install: bool = RUNTIME_DEPENDENCY_INSTALL,
) -> None:
    """Run Flower AgentApp process."""
    # Monitor the main process in case of SIGKILL
    if parent_pid is not None:
        start_parent_process_monitor(parent_pid)

    lifecycle, exit_code = _run_agentapp_task(
        runtime_api_address,
        log_queue,
        token,
        insecure,
        validate_and_resolve_root_certificates(certificates_path, insecure),
        certificates_path,
        runtime_dependency_install,
    )

    flwr_exit(
        code=exit_code,
        event_type=EventType.FLWR_AGENTAPP_RUN_LEAVE,
        event_details=lifecycle.event_details(exit_code),
    )


def _set_runtime_environment(
    runtime_api_address: str,
    token: str,
    insecure: bool,
    root_certificates_path: str | None,
) -> None:
    """Expose the Open Responses-compatible Runtime endpoint to the AgentApp."""
    scheme = "http" if insecure else "https"
    address = runtime_api_address.rstrip("/")
    os.environ[_RUNTIME_BASE_URL_ENV] = f"{scheme}://{address}/v1/runtime"
    os.environ[_RUNTIME_API_KEY_ENV] = token
    if root_certificates_path is not None:
        os.environ[_SSL_CERT_FILE_ENV] = root_certificates_path
