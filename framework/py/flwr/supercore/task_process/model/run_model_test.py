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
import os
import signal
import threading
from unittest.mock import Mock

import pytest

from flwr.common.constant import SubStatus
from flwr.proto.message_pb2 import Context as ProtoContext  # pylint: disable=E0611
from flwr.proto.run_pb2 import Run as ProtoRun  # pylint: disable=E0611
from flwr.proto.runtime_pb2 import PullTaskInputResponse  # pylint: disable=E0611
from flwr.supercore.exit import ExitCode
from flwr.supercore.task_identity import TaskIdentity
from flwr.supercore.telemetry import EventType

run_model_module = importlib.import_module(
    "flwr.supercore.task_process.model.run_model"
)


@pytest.mark.parametrize(
    ("failure", "returncode", "sub_status", "details"),
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
def test_run_model_once_cleans_up_fresh_task_state(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception | None,
    returncode: int,
    sub_status: SubStatus,
    details: str,
) -> None:
    """One invocation should own, report, and close its task-scoped state."""
    client = Mock()
    client.PullTaskInput.return_value = PullTaskInputResponse(
        task_id=17,
        run=ProtoRun(run_id=42),
        context=ProtoContext(node_id=99),
    )
    retry_invoker = Mock(max_tries=10)
    heartbeat = Mock(is_running=True)
    leave_future = Mock()
    leave_future.result.side_effect = TimeoutError
    telemetry = Mock(side_effect=[Mock(), leave_future])
    force_exit_timer = Mock()
    timer_cls = Mock(return_value=force_exit_timer)
    monkeypatch.setattr(
        run_model_module,
        "_create_runtime_client",
        Mock(return_value=(client, retry_invoker)),
    )
    monkeypatch.setattr(
        run_model_module, "HeartbeatSender", Mock(return_value=heartbeat)
    )
    monkeypatch.setattr(run_model_module, "handle_task", Mock(side_effect=failure))
    monkeypatch.setattr(run_model_module, "event", telemetry)
    monkeypatch.setattr(run_model_module, "_register_resident_signal_handlers", Mock())
    monkeypatch.setattr(run_model_module.threading, "Timer", timer_cls)

    assert (
        run_model_module.run_model_once("runtime.example:9092", "task-token", True)
        == returncode
    )

    assert (TaskIdentity.task_id, TaskIdentity.run_id, TaskIdentity.node_id) == (
        17,
        42,
        99,
    )
    output = client.PushTaskOutput.call_args.args[0]
    assert (output.sub_status, output.details) == (sub_status, details)
    assert retry_invoker.max_tries == 1
    heartbeat.stop.assert_called_once_with()
    client.close.assert_called_once_with()
    assert telemetry.call_args_list[-1].args == (
        EventType.FLWR_MODEL_RUN_LEAVE,
        {"exit_code": 0 if failure is None else ExitCode.TASK_PROC_EXCEPTION},
    )
    leave_future.result.assert_called_once_with(
        timeout=run_model_module.TELEMETRY_TIMEOUT_SECONDS
    )
    timer_cls.assert_called_once_with(
        run_model_module.FORCE_EXIT_TIMEOUT_SECONDS,
        os._exit,
        args=(returncode,),
    )
    assert force_exit_timer.daemon
    force_exit_timer.start.assert_called_once_with()
    force_exit_timer.cancel.assert_called_once_with()


def test_run_model_once_force_exits_if_cleanup_hangs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resident task should force exit if cleanup exceeds its timeout."""
    force_exit_called = threading.Event()
    lifecycle = Mock()
    lifecycle.complete.side_effect = lambda _: force_exit_called.wait(timeout=1.0)
    force_exit = Mock(side_effect=lambda _: force_exit_called.set())
    monkeypatch.setattr(
        run_model_module,
        "_run_model_task",
        Mock(return_value=(lifecycle, ExitCode.SUCCESS)),
    )
    monkeypatch.setattr(run_model_module, "FORCE_EXIT_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(run_model_module.os, "_exit", force_exit)

    assert run_model_module.run_model_once("runtime.example:9092", "token", True) == 0

    force_exit.assert_called_once_with(0)


def test_finalization_ignores_graceful_signals_process_wide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finalization should restore the process-wide graceful signal handlers."""
    previous_handlers = {sig: object() for sig in run_model_module.SIGNAL_TO_EXIT_CODE}
    handlers = previous_handlers.copy()

    def register(sig: int, handler: object) -> object:
        previous = handlers[sig]
        handlers[sig] = handler
        return previous

    monkeypatch.setattr(signal, "signal", register)

    with run_model_module._ignore_graceful_signals():
        assert all(handler == signal.SIG_IGN for handler in handlers.values())

    assert handlers == previous_handlers


@pytest.mark.parametrize(
    "recorded_failure_details",
    [None, "Model task failed with exception: provider failed"],
)
def test_resident_signal_finalization_is_exactly_once(
    monkeypatch: pytest.MonkeyPatch, recorded_failure_details: str | None
) -> None:
    """A resident signal should finalize once without replacing prior outcomes."""
    client = Mock()
    heartbeat = Mock(is_running=True)
    telemetry = Mock()
    monkeypatch.setattr(
        run_model_module,
        "_create_runtime_client",
        Mock(return_value=(client, Mock(max_tries=10))),
    )
    monkeypatch.setattr(run_model_module, "event", telemetry)
    lifecycle = run_model_module._ModelTaskLifecycle(
        "runtime.example:9092", "task-token", True, None
    )
    lifecycle.initialize()
    if recorded_failure_details is None:
        lifecycle._heartbeat_sender = heartbeat
    else:
        client.PullTaskInput.side_effect = RuntimeError("provider failed")
        monkeypatch.setattr(
            run_model_module, "HeartbeatSender", Mock(return_value=heartbeat)
        )
        assert lifecycle.run() == ExitCode.TASK_PROC_EXCEPTION
    handlers: dict[int, object] = {}
    exit_handlers: list[object] = []

    def register(sig: int, handler: object) -> object:
        previous = handlers.get(sig, signal.SIG_DFL)
        handlers[sig] = handler
        return previous

    monkeypatch.setattr(signal, "signal", register)
    monkeypatch.setattr(run_model_module, "add_exit_handler", exit_handlers.append)
    flwr_exit = Mock()
    monkeypatch.setattr(run_model_module, "flwr_exit", flwr_exit)

    run_model_module._register_resident_signal_handlers(lifecycle)
    handler = handlers[signal.SIGTERM]
    assert callable(handler)
    handler(signal.SIGTERM, None)

    assert len(exit_handlers) == 1
    exit_handler = exit_handlers[0]
    assert callable(exit_handler)
    threads = [threading.Thread(target=exit_handler) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1.0)
        assert not thread.is_alive()
    lifecycle.complete(ExitCode.SUCCESS)

    output = client.PushTaskOutput.call_args.args[0]
    assert (output.sub_status, output.details) == (
        SubStatus.FAILED,
        recorded_failure_details or "Model task stopped by user.",
    )
    heartbeat.stop.assert_called_once_with()
    client.close.assert_called_once_with()
    telemetry.assert_called_once_with(
        EventType.FLWR_MODEL_RUN_LEAVE,
        {"exit_code": ExitCode.GRACEFUL_EXIT_SIGTERM},
    )
    telemetry.return_value.result.assert_called_once_with(
        timeout=run_model_module.TELEMETRY_TIMEOUT_SECONDS
    )
    flwr_exit.assert_called_once_with(
        ExitCode.GRACEFUL_EXIT_SIGTERM,
        message="Run stopped by user.",
        emit_telemetry=False,
    )


def test_resident_accepts_only_after_initialization_and_signal_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task authority should move only after resident state is ready."""
    calls: list[str] = []
    lifecycle = Mock()

    def run() -> int:
        calls.append("run")
        return ExitCode.SUCCESS

    lifecycle.initialize.side_effect = lambda: calls.append("initialize")
    lifecycle.run.side_effect = run
    monkeypatch.setattr(
        run_model_module, "_ModelTaskLifecycle", Mock(return_value=lifecycle)
    )
    monkeypatch.setattr(
        run_model_module,
        "_register_resident_signal_handlers",
        lambda _: calls.append("signals"),
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
    assert (returned_lifecycle, exit_code) == (lifecycle, ExitCode.SUCCESS)


def test_cold_model_keeps_flower_exit_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cold CLI should keep Flower exit behavior around the shared task."""
    lifecycle = Mock()
    run_task = Mock(return_value=(lifecycle, ExitCode.TASK_PROC_EXCEPTION))
    flwr_exit = Mock()
    monkeypatch.setattr(run_model_module, "_run_model_task", run_task)
    monkeypatch.setattr(run_model_module, "flwr_exit", flwr_exit)

    run_model_module.run_model(
        "runtime.example:9092", "task-token", True, certificates=b"ca"
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
