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
"""Flower model task process."""


from __future__ import annotations

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
from flwr.supercore.constant import TELEMETRY_TIMEOUT_SECONDS
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
from flwr.supercore.telemetry import EventType, event

from .task import handle_task


class _ModelTaskLifecycle:  # pylint: disable=too-many-instance-attributes
    """Own the task-scoped Model state and its exactly-once finalization."""

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
        self._details = "Model task failed with unknown error."
        self._lock = threading.RLock()
        self._finalized = False
        self._leave_event_started = False

    def run(self) -> int:
        """Execute the task and return its Flower exit code."""
        exit_code = ExitCode.SUCCESS
        try:
            if self._client is None:
                raise RuntimeError("Model Runtime client initialization failed.")
            self._heartbeat_sender = HeartbeatSender(
                make_task_heartbeat_fn_http(self._client)
            )
            self._heartbeat_sender.start()

            log(DEBUG, "[flwr-model] Pull task input")
            task_input: PullTaskInputResponse = self._client.PullTaskInput(
                PullTaskInputRequest()
            )

            event(EventType.FLWR_MODEL_RUN_ENTER)
            handle_task(
                client=self._client,
                task_id=task_input.task_id,
                run_id=task_input.run.run_id,
            )

            with self._lock:
                self._sub_status = SubStatus.COMPLETED
                self._details = ""
        except Exception as ex:  # pylint: disable=broad-exception-caught
            log(ERROR, "`flwr-model` failed", exc_info=ex)
            with self._lock:
                self._sub_status = SubStatus.FAILED
                self._details = f"Model task failed with exception: {str(ex)}"
            exit_code = ExitCode.TASK_PROC_EXCEPTION
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
            if self._finalized or self._sub_status == SubStatus.COMPLETED:
                return
            self._sub_status = SubStatus.FAILED
            self._details = "Model task stopped by user."

    def finalize(self) -> None:
        """Push final status and release task state exactly once."""
        with _defer_graceful_signals(), self._lock:
            if self._finalized:
                return
            self._finalized = True

            log(DEBUG, "[flwr-model] Will push Model task output")
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
                log(ERROR, "Failed to push task output: %s", str(err))

            try:
                if self._heartbeat_sender and self._heartbeat_sender.is_running:
                    self._heartbeat_sender.stop()
            except Exception as err:  # pylint: disable=broad-exception-caught
                log(ERROR, "Failed to stop Model task heartbeat", exc_info=err)
            try:
                self._client.close()
            except Exception as err:  # pylint: disable=broad-exception-caught
                log(ERROR, "Failed to close Model Runtime client", exc_info=err)

    def complete(self, exit_code: int) -> None:
        """Finalize and emit one bounded leave event for a resident worker."""
        with _defer_graceful_signals():
            self.finalize()
            with self._lock:
                if self._leave_event_started:
                    return
                self._leave_event_started = True
            future: Future[str] = event(
                EventType.FLWR_MODEL_RUN_LEAVE, {"exit_code": exit_code}
            )
            try:
                future.result(timeout=TELEMETRY_TIMEOUT_SECONDS)
            except Exception:  # pylint: disable=broad-exception-caught
                pass


@contextmanager
def _defer_graceful_signals() -> Iterator[None]:
    """Defer graceful signals while exactly-once finalization is in progress."""
    pthread_sigmask = getattr(signal, "pthread_sigmask", None)
    if not callable(pthread_sigmask) or (
        threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    previous_mask = pthread_sigmask(signal.SIG_BLOCK, set(SIGNAL_TO_EXIT_CODE))
    try:
        yield
    finally:
        pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _run_model_task(  # pylint: disable=too-many-arguments
    runtime_api_address: str,
    token: str,
    insecure: bool,
    certificates: bytes | None,
    *,
    resident: bool,
    on_started: Callable[[], None] | None = None,
) -> tuple[_ModelTaskLifecycle, int]:
    """Create and execute one task through the shared Model lifecycle."""
    lifecycle = _ModelTaskLifecycle(
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
            event_type=EventType.FLWR_MODEL_RUN_LEAVE,
            exit_message="Run stopped by user.",
            exit_handlers=[lifecycle.finalize],
        )
    if on_started is not None:
        on_started()
    return lifecycle, lifecycle.run()


def _register_resident_signal_handlers(lifecycle: _ModelTaskLifecycle) -> None:
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
        # Let `flwr_exit` start its force-exit timer before cleanup. The
        # lifecycle itself bounds the telemetry wait and is idempotent if
        # normal completion raced with this signal.
        add_exit_handler(lambda: lifecycle.complete(exit_code))
        flwr_exit(
            exit_code,
            message="Run stopped by user.",
            emit_telemetry=False,
        )

    for sig in SIGNAL_TO_EXIT_CODE:
        default_handlers[sig] = signal.signal(sig, graceful_exit_handler)


def run_model_once(
    runtime_api_address: str,
    token: str,
    insecure: bool,
    certificates: bytes | None = None,
    on_started: Callable[[], None] | None = None,
) -> int:
    """Run one Model task without terminating the containing process."""
    lifecycle, exit_code = _run_model_task(
        runtime_api_address,
        token,
        insecure,
        certificates,
        resident=True,
        on_started=on_started,
    )
    lifecycle.complete(exit_code)
    return 0 if exit_code == ExitCode.SUCCESS else 1


def run_model(
    runtime_api_address: str,
    token: str,
    insecure: bool,
    certificates: bytes | None = None,
    parent_pid: int | None = None,
) -> None:
    """Run Flower model task process."""
    # Monitor the main process in case of SIGKILL
    if parent_pid is not None:
        start_parent_process_monitor(parent_pid)

    _, exit_code = _run_model_task(
        runtime_api_address,
        token,
        insecure,
        certificates,
        resident=False,
    )
    flwr_exit(exit_code, event_type=EventType.FLWR_MODEL_RUN_LEAVE)


def _create_runtime_client(
    *,
    runtime_api_address: str,
    token: str,
    insecure: bool,
    certificates: bytes | None,
) -> tuple[RuntimeHttpClient, RetryInvoker]:
    """Create a Runtime HTTP client authenticated as the model task."""
    retry_invoker = make_simple_http_retry_invoker()
    client = RuntimeHttpClient.from_server_address(
        server_address=runtime_api_address,
        insecure=insecure,
        root_certificates=certificates,
        interceptors=[
            RuntimeVersionHttpInterceptor(component_name="flwr-model"),
            RuntimeTokenHttpInterceptor(token),
        ],
        retry_invoker=retry_invoker,
    )
    return client, retry_invoker
