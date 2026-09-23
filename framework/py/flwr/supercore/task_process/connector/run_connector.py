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

import os
import signal
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from logging import DEBUG, ERROR
from types import FrameType
from typing import Any

from flwr.common.constant import SubStatus
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    PullTaskInputRequest,
    PullTaskInputResponse,
    PushTaskOutputRequest,
)
from flwr.supercore import log
from flwr.supercore.app_utils import start_parent_process_monitor
from flwr.supercore.constant import (
    FORCE_EXIT_TIMEOUT_SECONDS,
    TELEMETRY_TIMEOUT_SECONDS,
)
from flwr.supercore.exit import (
    ExitCode,
    add_exit_handler,
    flwr_exit,
    register_signal_handlers,
)
from flwr.supercore.exit.signal_handler import SIGNAL_TO_EXIT_CODE
from flwr.supercore.heartbeat import HeartbeatSender, make_task_heartbeat_fn_http
from flwr.supercore.interceptors import (
    RuntimeTokenHttpInterceptor,
    RuntimeVersionHttpInterceptor,
)
from flwr.supercore.retry import RetryInvoker, make_simple_http_retry_invoker
from flwr.supercore.runtime import RuntimeHttpClient
from flwr.supercore.task_identity import TaskIdentity
from flwr.supercore.telemetry import EventType, event

from .task import handle_task


class _ConnectorTaskLifecycle:  # pylint: disable=too-many-instance-attributes
    """Own task-scoped Connector state and its exactly-once finalization."""

    def __init__(
        self,
        runtime_api_address: str,
        token: str,
        insecure: bool,
        certificates: bytes | None,
    ) -> None:
        self._runtime_api_address = runtime_api_address
        self._token = token
        self._insecure = insecure
        self._certificates = certificates
        self._client: RuntimeHttpClient | None = None
        self._retry_invoker: RetryInvoker | None = None
        self._heartbeat_sender: HeartbeatSender | None = None
        self._sub_status = SubStatus.FAILED
        self._details = "Connector task failed with unknown error."
        self._lock = threading.RLock()
        self._finalized = False
        self._task_finished = False
        self._leave_event_started = False

    def run(self) -> int:
        """Execute the task and return its Flower exit code."""
        exit_code = ExitCode.SUCCESS
        try:
            if self._client is None:
                raise RuntimeError("Connector Runtime client initialization failed.")
            self._heartbeat_sender = HeartbeatSender(
                make_task_heartbeat_fn_http(self._client)
            )
            self._heartbeat_sender.start()

            log(DEBUG, "[flwr-connector] Pull task input")
            task_input: PullTaskInputResponse = self._client.PullTaskInput(
                PullTaskInputRequest()
            )
            TaskIdentity.task_id = task_input.task_id
            TaskIdentity.run_id = task_input.run.run_id
            TaskIdentity.node_id = task_input.context.node_id

            event(EventType.FLWR_CONNECTOR_RUN_ENTER)
            handle_task(client=self._client)

            with self._lock:
                self._sub_status = SubStatus.COMPLETED
                self._details = ""
                self._task_finished = True
        except Exception as ex:  # pylint: disable=broad-exception-caught
            log(ERROR, "`flwr-connector` failed", exc_info=ex)
            with self._lock:
                self._sub_status = SubStatus.FAILED
                self._details = f"Connector task failed with exception: {str(ex)}"
                self._task_finished = True
            exit_code = (
                ExitCode.COMMON_APP_IMPORT_ERROR
                if isinstance(ex, ImportError)
                else ExitCode.TASK_PROC_EXCEPTION
            )
        return exit_code

    def initialize(self) -> None:
        """Create fresh task-scoped Runtime state if it is not initialized."""
        if self._client is not None:
            return
        self._client, self._retry_invoker = _create_runtime_client(
            runtime_api_address=self._runtime_api_address,
            token=self._token,
            insecure=self._insecure,
            certificates=self._certificates,
        )

    def mark_interrupted(self) -> None:
        """Record a graceful interruption before final task output is pushed."""
        with self._lock:
            if self._finalized or self._task_finished:
                return
            self._sub_status = SubStatus.FAILED
            self._details = "Connector task stopped by user."

    def finalize(self) -> None:
        """Push final status and release task state exactly once."""
        with _ignore_graceful_signals(), self._lock:
            if self._finalized:
                return
            self._finalized = True

            log(DEBUG, "[flwr-connector] Will push Connector task output")
            if self._client is None or self._retry_invoker is None:
                return
            self._retry_invoker.max_tries = 1
            try:
                self._client.PushTaskOutput(
                    PushTaskOutputRequest(
                        sub_status=self._sub_status,
                        details=self._details,
                    )
                )
            except Exception as err:  # pylint: disable=broad-exception-caught
                log(ERROR, "Failed to push task output", exc_info=err)

            try:
                if self._heartbeat_sender and self._heartbeat_sender.is_running:
                    self._heartbeat_sender.stop()
            except Exception as err:  # pylint: disable=broad-exception-caught
                log(ERROR, "Failed to stop Connector task heartbeat", exc_info=err)
            try:
                self._client.close()
            except Exception as err:  # pylint: disable=broad-exception-caught
                log(ERROR, "Failed to close Connector Runtime client", exc_info=err)

    def complete(self, exit_code: int) -> None:
        """Finalize and emit one bounded leave event for a resident worker."""
        with _ignore_graceful_signals():
            self.finalize()
            with self._lock:
                if self._leave_event_started:
                    return
                self._leave_event_started = True
            try:
                future: Future[str] = event(
                    EventType.FLWR_CONNECTOR_RUN_LEAVE, {"exit_code": exit_code}
                )
                future.result(timeout=TELEMETRY_TIMEOUT_SECONDS)
            except Exception:  # pylint: disable=broad-exception-caught
                pass


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


def _run_connector_task(  # pylint: disable=too-many-arguments
    runtime_api_address: str,
    token: str,
    insecure: bool,
    certificates: bytes | None,
    *,
    resident: bool,
    on_started: Callable[[], None] | None = None,
) -> tuple[_ConnectorTaskLifecycle, int]:
    """Create and execute one task through the Connector lifecycle."""
    lifecycle = _ConnectorTaskLifecycle(
        runtime_api_address,
        token,
        insecure,
        certificates,
    )
    lifecycle.initialize()
    if resident:
        _register_resident_signal_handlers(lifecycle)
    else:
        # Preserve the cold path's existing ordering: create its Runtime client
        # before installing task signal handlers.
        register_signal_handlers(
            event_type=EventType.FLWR_CONNECTOR_RUN_LEAVE,
            exit_message="Task stopped by user.",
            exit_handlers=[lifecycle.finalize],
        )
    # Transfer task authority only after Runtime state and signal ownership exist.
    if on_started is not None:
        on_started()
    return lifecycle, lifecycle.run()


def _register_resident_signal_handlers(lifecycle: _ConnectorTaskLifecycle) -> None:
    """Register coordinated graceful exits for the resident PID 1 worker."""
    default_handlers: dict[int, Any] = {}
    is_exiting = False
    lock = threading.Lock()

    def graceful_exit_handler(signalnum: int, _frame: FrameType | None) -> None:
        nonlocal is_exiting
        with lock:
            if is_exiting:
                return
            is_exiting = True

        for sig, default_handler in default_handlers.items():
            signal.signal(sig, default_handler)

        exit_code = SIGNAL_TO_EXIT_CODE[signalnum]
        lifecycle.mark_interrupted()
        add_exit_handler(lambda: lifecycle.complete(exit_code))
        flwr_exit(
            exit_code,
            message="Task stopped by user.",
            emit_telemetry=False,
        )

    for sig in SIGNAL_TO_EXIT_CODE:
        default_handlers[sig] = signal.signal(sig, graceful_exit_handler)


def run_connector_once(
    runtime_api_address: str,
    token: str,
    insecure: bool,
    certificates: bytes | None = None,
    on_started: Callable[[], None] | None = None,
) -> int:
    """Run one Connector task without terminating the containing process."""
    lifecycle, exit_code = _run_connector_task(
        runtime_api_address,
        token,
        insecure,
        certificates,
        resident=True,
        on_started=on_started,
    )
    returncode = 0 if exit_code == ExitCode.SUCCESS else 1
    force_exit_timer = threading.Timer(
        FORCE_EXIT_TIMEOUT_SECONDS,
        os._exit,
        args=(returncode,),
    )
    force_exit_timer.daemon = True
    force_exit_timer.start()
    try:
        lifecycle.complete(exit_code)
    finally:
        force_exit_timer.cancel()
    return returncode


def run_connector(
    runtime_api_address: str,
    token: str,
    insecure: bool,
    certificates: bytes | None = None,
    parent_pid: int | None = None,
) -> None:
    """Run Flower connector task process."""
    if parent_pid is not None:
        start_parent_process_monitor(parent_pid)

    _, exit_code = _run_connector_task(
        runtime_api_address,
        token,
        insecure,
        certificates,
        resident=False,
    )
    flwr_exit(exit_code, event_type=EventType.FLWR_CONNECTOR_RUN_LEAVE)


def _create_runtime_client(
    *,
    runtime_api_address: str,
    token: str,
    insecure: bool,
    certificates: bytes | None,
) -> tuple[RuntimeHttpClient, RetryInvoker]:
    """Create a Runtime HTTP client authenticated as the connector task."""
    retry_invoker = make_simple_http_retry_invoker()
    client = RuntimeHttpClient.from_server_address(
        server_address=runtime_api_address,
        insecure=insecure,
        root_certificates=certificates,
        interceptors=[
            RuntimeVersionHttpInterceptor(component_name="flwr-connector"),
            RuntimeTokenHttpInterceptor(token),
        ],
        retry_invoker=retry_invoker,
    )
    return client, retry_invoker
