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
"""Tests for one-task Model execution inside a prestarted worker."""

# pylint: disable=protected-access

import importlib
import multiprocessing
import os
import signal
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from flwr.common.constant import SubStatus
from flwr.proto.message_pb2 import Context as ProtoContext  # pylint: disable=E0611
from flwr.proto.run_pb2 import Run as ProtoRun  # pylint: disable=E0611
from flwr.proto.runtime_pb2 import PullTaskInputResponse  # pylint: disable=E0611
from flwr.supercore.exit import ExitCode
from flwr.supercore.exit.exit_handler import registered_exit_handlers
from flwr.supercore.task_identity import TaskIdentity
from flwr.supercore.telemetry import EventType

run_model_module = importlib.import_module(
    "flwr.supercore.task_process.model.run_model"
)


@pytest.mark.parametrize(
    ("failure", "expected_returncode", "expected_status", "expected_details"),
    [
        (None, 0, SubStatus.COMPLETED, ""),
        (
            RuntimeError("provider failed"),
            1,
            SubStatus.FAILED,
            "Model task failed with exception: provider failed",
        ),
    ],
)
def test_run_model_once_uses_fresh_task_scoped_state_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception | None,
    expected_returncode: int,
    expected_status: SubStatus,
    expected_details: str,
) -> None:
    """One invocation should own and close its Runtime and heartbeat state."""
    client = Mock()
    client.PullTaskInput.return_value = PullTaskInputResponse(
        task_id=17,
        run=ProtoRun(run_id=42),
        context=ProtoContext(node_id=99),
    )
    retry_invoker = Mock(max_tries=10)
    create_client = Mock(return_value=(client, retry_invoker))
    heartbeat_sender = Mock(is_running=True)
    heartbeat_cls = Mock(return_value=heartbeat_sender)
    handle_task = Mock(side_effect=failure)
    telemetry_event = Mock()
    leave_future = Mock()
    telemetry_event.side_effect = [Mock(), leave_future]
    monkeypatch.setattr(run_model_module, "_create_runtime_client", create_client)
    monkeypatch.setattr(run_model_module, "HeartbeatSender", heartbeat_cls)
    monkeypatch.setattr(run_model_module, "handle_task", handle_task)
    monkeypatch.setattr(run_model_module, "event", telemetry_event)
    monkeypatch.setattr(run_model_module, "_register_resident_signal_handlers", Mock())

    returncode = run_model_module.run_model_once(
        "runtime.example:9092", "task-token", True
    )

    assert returncode == expected_returncode
    create_client.assert_called_once_with(
        runtime_api_address="runtime.example:9092",
        token="task-token",
        insecure=True,
        certificates=None,
    )
    heartbeat_sender.start.assert_called_once_with()
    handle_task.assert_called_once_with(client=client)
    assert TaskIdentity.task_id == 17
    assert TaskIdentity.run_id == 42
    assert TaskIdentity.node_id == 99
    output = client.PushTaskOutput.call_args.args[0]
    assert output.sub_status == expected_status
    assert output.details == expected_details
    assert retry_invoker.max_tries == 1
    heartbeat_sender.stop.assert_called_once_with()
    client.close.assert_called_once_with()
    assert telemetry_event.call_args_list[0].args == (EventType.FLWR_MODEL_RUN_ENTER,)
    assert telemetry_event.call_args_list[-1].args == (
        EventType.FLWR_MODEL_RUN_LEAVE,
        {"exit_code": 0 if failure is None else 800},
    )
    leave_future.result.assert_called_once_with(
        timeout=run_model_module.TELEMETRY_TIMEOUT_SECONDS
    )


def test_model_task_finalization_is_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent completion paths should release task-scoped state once."""
    client = Mock()
    retry_invoker = Mock(max_tries=10)
    heartbeat_sender = Mock(is_running=True)
    monkeypatch.setattr(
        run_model_module,
        "_create_runtime_client",
        Mock(return_value=(client, retry_invoker)),
    )
    monkeypatch.setattr(
        run_model_module, "HeartbeatSender", Mock(return_value=heartbeat_sender)
    )

    lifecycle = run_model_module._ModelTaskLifecycle(
        "runtime.example:9092", "task-token", True, None
    )
    lifecycle.initialize()
    lifecycle._heartbeat_sender = heartbeat_sender
    threads = [threading.Thread(target=lifecycle.finalize) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1.0)

    client.PushTaskOutput.assert_called_once()
    heartbeat_sender.stop.assert_called_once_with()
    client.close.assert_called_once_with()
    assert retry_invoker.max_tries == 1


def test_resident_signal_finalizes_and_exits_with_flower_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGTERM should finalize one accepted task before terminating the worker."""
    handlers: dict[int, object] = {}

    def register(sig: int, handler: object) -> object:
        previous = handlers.get(sig, signal.SIG_DFL)
        handlers[sig] = handler
        return previous

    lifecycle = Mock()
    flwr_exit = Mock()
    add_exit_handler = Mock()
    monkeypatch.setattr(signal, "signal", register)
    monkeypatch.setattr(run_model_module, "flwr_exit", flwr_exit)
    monkeypatch.setattr(run_model_module, "add_exit_handler", add_exit_handler)

    run_model_module._register_resident_signal_handlers(lifecycle)
    handler = handlers[signal.SIGTERM]
    assert callable(handler)
    handler(signal.SIGTERM, None)

    lifecycle.mark_interrupted.assert_called_once_with()
    lifecycle.complete.assert_not_called()
    exit_handler = add_exit_handler.call_args.args[0]
    exit_handler()
    lifecycle.complete.assert_called_once_with(ExitCode.GRACEFUL_EXIT_SIGTERM)
    flwr_exit.assert_called_once_with(
        ExitCode.GRACEFUL_EXIT_SIGTERM,
        message="Run stopped by user.",
        emit_telemetry=False,
    )


def test_sigterm_completion_releases_accepted_task_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Signal completion should publish failure and clean up task state once."""
    client = Mock()
    retry_invoker = Mock(max_tries=10)
    heartbeat_sender = Mock(is_running=True)
    future = Mock()
    telemetry_event = Mock(return_value=future)
    monkeypatch.setattr(
        run_model_module,
        "_create_runtime_client",
        Mock(return_value=(client, retry_invoker)),
    )
    monkeypatch.setattr(run_model_module, "event", telemetry_event)
    lifecycle = run_model_module._ModelTaskLifecycle(
        "runtime.example:9092", "task-token", True, None
    )
    lifecycle.initialize()
    lifecycle._heartbeat_sender = heartbeat_sender

    lifecycle.mark_interrupted()
    lifecycle.complete(ExitCode.GRACEFUL_EXIT_SIGTERM)
    lifecycle.complete(ExitCode.GRACEFUL_EXIT_SIGTERM)

    output = client.PushTaskOutput.call_args.args[0]
    assert output.sub_status == SubStatus.FAILED
    assert output.details == "Model task stopped by user."
    heartbeat_sender.stop.assert_called_once_with()
    client.close.assert_called_once_with()
    telemetry_event.assert_called_once_with(
        EventType.FLWR_MODEL_RUN_LEAVE,
        {"exit_code": ExitCode.GRACEFUL_EXIT_SIGTERM},
    )
    future.result.assert_called_once_with(
        timeout=run_model_module.TELEMETRY_TIMEOUT_SECONDS
    )


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX signals")
def test_sigterm_during_accepted_task_terminates_resident_worker() -> None:
    """An accepted resident task should finalize before SIGTERM exits PID 1."""
    context = multiprocessing.get_context("fork")
    accepted = context.Event()
    task_running = context.Event()
    read_events, write_events = context.Pipe(duplex=False)

    def child() -> None:
        """Run one blocking resident task in a signal-isolated child."""
        registered_exit_handlers.clear()

        class Client:
            """Record task lifecycle calls across the process boundary."""

            def PullTaskInput(  # pylint: disable=invalid-name
                self, _request: object
            ) -> PullTaskInputResponse:
                """Return one fixed task input."""
                return PullTaskInputResponse(
                    task_id=17,
                    run=ProtoRun(run_id=42),
                )

            def PushTaskOutput(  # pylint: disable=invalid-name
                self, request: Any
            ) -> None:
                """Record the final task output."""
                write_events.send(
                    (
                        "output",
                        request.sub_status,
                        request.details,
                    )
                )

            def close(self) -> None:
                """Record Runtime client closure."""
                write_events.send(("client_closed",))

        class Heartbeat:
            """Record heartbeat cleanup across the process boundary."""

            is_running = True

            def start(self) -> None:
                """Record heartbeat startup and expose the running boundary."""
                write_events.send(("heartbeat_started",))
                task_running.set()

            def stop(self) -> None:
                """Record heartbeat shutdown."""
                write_events.send(("heartbeat_stopped",))

        def telemetry(event_type: EventType, details: Any = None) -> Future[str]:
            """Record one immediately completed telemetry event."""
            write_events.send(("telemetry", event_type, details))
            future: Future[str] = Future()
            future.set_result("sent")
            return future

        def block_task(**_kwargs: object) -> None:
            """Keep the accepted task active until the process receives a signal."""
            while True:
                time.sleep(0.1)

        with (
            patch.object(
                run_model_module,
                "_create_runtime_client",
                Mock(return_value=(Client(), SimpleNamespace(max_tries=10))),
            ),
            patch.object(
                run_model_module,
                "HeartbeatSender",
                Mock(return_value=Heartbeat()),
            ),
            patch.object(
                run_model_module,
                "make_task_heartbeat_fn_http",
                Mock(return_value=Mock()),
            ),
            patch.object(run_model_module, "handle_task", block_task),
            patch.object(run_model_module, "event", telemetry),
        ):
            run_model_module.run_model_once(
                "runtime.example:9092",
                "task-token",
                True,
                on_started=accepted.set,
            )

    process = context.Process(target=child)
    process.start()
    write_events.close()
    try:
        assert accepted.wait(timeout=2.0)
        assert task_running.wait(timeout=2.0)
        assert process.pid is not None
        os.kill(process.pid, signal.SIGTERM)
        process.join(timeout=5.0)
        assert not process.is_alive()
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)

    events: list[tuple[Any, ...]] = []
    while read_events.poll():
        try:
            events.append(read_events.recv())
        except EOFError:
            break
    read_events.close()

    assert process.exitcode == 0
    assert ("heartbeat_stopped",) in events
    assert ("client_closed",) in events
    output = next(event for event in events if event[0] == "output")
    assert output[1:] == (SubStatus.FAILED, "Model task stopped by user.")
    leave_events = [
        event
        for event in events
        if event[:2] == ("telemetry", EventType.FLWR_MODEL_RUN_LEAVE)
    ]
    assert leave_events == [
        (
            "telemetry",
            EventType.FLWR_MODEL_RUN_LEAVE,
            {"exit_code": ExitCode.GRACEFUL_EXIT_SIGTERM},
        )
    ]


def test_resident_initializes_and_owns_signals_before_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker should own Runtime state and signals before acknowledging."""
    calls: list[str] = []
    lifecycle = Mock()

    def initialize() -> None:
        calls.append("initialize")

    def run() -> int:
        calls.append("run")
        return ExitCode.SUCCESS

    def register_signals(_lifecycle: object) -> None:
        calls.append("signals")

    lifecycle.initialize.side_effect = initialize
    lifecycle.run.side_effect = run
    lifecycle_cls = Mock(return_value=lifecycle)
    register = Mock(side_effect=register_signals)
    monkeypatch.setattr(run_model_module, "_ModelTaskLifecycle", lifecycle_cls)
    monkeypatch.setattr(
        run_model_module, "_register_resident_signal_handlers", register
    )

    returned_lifecycle, exit_code = run_model_module._run_model_task(
        "runtime.example:9092",
        "task-token",
        True,
        None,
        resident=True,
        on_started=lambda: calls.append("accepted"),
    )

    assert calls == ["initialize", "signals", "accepted", "run"]
    assert returned_lifecycle is lifecycle
    assert exit_code == ExitCode.SUCCESS


def test_resident_completion_bounds_telemetry_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resident completion should use the same bounded telemetry wait as exit."""
    client = Mock()
    future = Mock()
    future.result.side_effect = TimeoutError
    monkeypatch.setattr(
        run_model_module,
        "_create_runtime_client",
        Mock(return_value=(client, Mock(max_tries=10))),
    )
    monkeypatch.setattr(run_model_module, "event", Mock(return_value=future))
    lifecycle = run_model_module._ModelTaskLifecycle(
        "runtime.example:9092", "task-token", True, None
    )
    lifecycle.initialize()

    lifecycle.complete(ExitCode.GRACEFUL_EXIT_SIGTERM)
    lifecycle.complete(ExitCode.SUCCESS)

    future.result.assert_called_once_with(
        timeout=run_model_module.TELEMETRY_TIMEOUT_SECONDS
    )
    client.PushTaskOutput.assert_called_once()
    client.close.assert_called_once_with()


def test_cold_model_uses_shared_lifecycle_and_flwr_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cold CLI should retain Flower exit behavior around the shared task."""
    lifecycle = Mock()
    run_task = Mock(return_value=(lifecycle, ExitCode.TASK_PROC_EXCEPTION))
    flwr_exit = Mock()
    monkeypatch.setattr(run_model_module, "_run_model_task", run_task)
    monkeypatch.setattr(run_model_module, "flwr_exit", flwr_exit)

    run_model_module.run_model(
        "runtime.example:9092",
        "task-token",
        True,
        certificates=b"ca",
    )

    run_task.assert_called_once_with(
        "runtime.example:9092",
        "task-token",
        True,
        b"ca",
        resident=False,
    )
    flwr_exit.assert_called_once_with(
        ExitCode.TASK_PROC_EXCEPTION,
        event_type=EventType.FLWR_MODEL_RUN_LEAVE,
    )
