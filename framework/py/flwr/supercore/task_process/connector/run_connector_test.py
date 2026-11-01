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
"""Tests for one-task Connector execution inside a prestarted worker."""

# pylint: disable=protected-access

import importlib
import os
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

run_connector_module = importlib.import_module(
    "flwr.supercore.task_process.connector.run_connector"
)


@pytest.mark.parametrize(
    ("failure", "returncode", "exit_code", "sub_status", "details"),
    [
        (None, 0, ExitCode.SUCCESS, SubStatus.COMPLETED, ""),
        (
            RuntimeError("provider failed"),
            1,
            ExitCode.TASK_PROC_EXCEPTION,
            SubStatus.FAILED,
            "Connector task failed with exception: provider failed",
        ),
        (
            ImportError("connector missing"),
            1,
            ExitCode.COMMON_APP_IMPORT_ERROR,
            SubStatus.FAILED,
            "Connector task failed with exception: connector missing",
        ),
    ],
)
def test_run_connector_once_cleans_up_fresh_task_state(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception | None,
    returncode: int,
    exit_code: int,
    sub_status: SubStatus,
    details: str,
) -> None:
    """One invocation should own, report, and close its task-scoped state."""
    client = Mock()
    client.PullTaskInput.return_value = PullTaskInputResponse(
        task_id=23,
        run=ProtoRun(run_id=48),
        context=ProtoContext(node_id=101),
    )
    retry_invoker = Mock(max_tries=10)
    heartbeat = Mock(is_running=True)
    leave_future = Mock()
    leave_future.result.side_effect = TimeoutError
    telemetry = Mock(side_effect=[Mock(), leave_future])
    force_exit_timer = Mock()
    timer_cls = Mock(return_value=force_exit_timer)
    monkeypatch.setattr(
        run_connector_module,
        "_create_runtime_client",
        Mock(return_value=(client, retry_invoker)),
    )
    monkeypatch.setattr(
        run_connector_module, "HeartbeatSender", Mock(return_value=heartbeat)
    )
    monkeypatch.setattr(run_connector_module, "handle_task", Mock(side_effect=failure))
    monkeypatch.setattr(run_connector_module, "event", telemetry)
    monkeypatch.setattr(
        run_connector_module, "_register_resident_signal_handlers", Mock()
    )
    monkeypatch.setattr(run_connector_module.threading, "Timer", timer_cls)

    assert (
        run_connector_module.run_connector_once(
            "runtime.example:9092", "task-token", True
        )
        == returncode
    )

    assert (TaskIdentity.task_id, TaskIdentity.run_id, TaskIdentity.node_id) == (
        23,
        48,
        101,
    )
    output = client.PushTaskOutput.call_args.args[0]
    assert (output.sub_status, output.details) == (sub_status, details)
    assert retry_invoker.max_tries == 1
    heartbeat.stop.assert_called_once_with()
    client.close.assert_called_once_with()
    assert telemetry.call_args_list[-1].args == (
        EventType.FLWR_CONNECTOR_RUN_LEAVE,
        {"exit_code": exit_code},
    )
    leave_future.result.assert_called_once_with(
        timeout=run_connector_module.TELEMETRY_TIMEOUT_SECONDS
    )
    timer_cls.assert_called_once_with(
        run_connector_module.FORCE_EXIT_TIMEOUT_SECONDS,
        os._exit,
        args=(returncode,),
    )
    assert force_exit_timer.daemon
    force_exit_timer.start.assert_called_once_with()
    force_exit_timer.cancel.assert_called_once_with()


def test_resident_accepts_before_task_and_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task authority should move before input or credentials are requested."""
    calls: list[str] = []
    accepted = False
    client = Mock()

    def create_client(**_: object) -> tuple[Mock, Mock]:
        calls.append("client")
        return client, Mock(max_tries=10)

    def accept() -> None:
        nonlocal accepted
        assert not client.PullTaskInput.called
        assert not client.GetConnector.called
        accepted = True
        calls.append("accepted")

    def pull(_: object) -> PullTaskInputResponse:
        assert accepted
        calls.append("input")
        return PullTaskInputResponse(
            task_id=23,
            run=ProtoRun(run_id=48),
            context=ProtoContext(node_id=101),
        )

    def handle_task(*, client: Mock) -> None:
        assert accepted
        calls.append("credentials")
        client.GetConnector()

    client.PullTaskInput.side_effect = pull
    monkeypatch.setattr(run_connector_module, "_create_runtime_client", create_client)
    monkeypatch.setattr(run_connector_module, "HeartbeatSender", Mock())
    monkeypatch.setattr(run_connector_module, "event", Mock())
    monkeypatch.setattr(run_connector_module, "handle_task", handle_task)
    monkeypatch.setattr(
        run_connector_module,
        "_register_resident_signal_handlers",
        lambda _: calls.append("signals"),
    )

    lifecycle, exit_code = run_connector_module._run_connector_task(
        "runtime.example:9092",
        "task-token",
        True,
        None,
        resident=True,
        on_started=accept,
    )

    assert calls == ["client", "signals", "accepted", "input", "credentials"]
    assert exit_code == ExitCode.SUCCESS
    client.GetConnector.assert_called_once_with()
    lifecycle.finalize()


def test_connector_lifecycle_finalizes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated cleanup should push status and close task state only once."""
    client = Mock()
    retry_invoker = Mock(max_tries=10)
    monkeypatch.setattr(
        run_connector_module,
        "_create_runtime_client",
        Mock(return_value=(client, retry_invoker)),
    )
    lifecycle = run_connector_module._ConnectorTaskLifecycle(
        "runtime.example:9092", "task-token", True, None
    )
    lifecycle.initialize()

    lifecycle.finalize()
    lifecycle.finalize()

    client.PushTaskOutput.assert_called_once()
    client.close.assert_called_once_with()
    assert retry_invoker.max_tries == 1


def test_run_connector_once_force_exits_if_cleanup_hangs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resident task should force exit if cleanup exceeds its timeout."""
    force_exit_called = threading.Event()
    lifecycle = Mock()
    lifecycle.complete.side_effect = lambda _: force_exit_called.wait(timeout=1.0)
    force_exit = Mock(side_effect=lambda _: force_exit_called.set())
    monkeypatch.setattr(
        run_connector_module,
        "_run_connector_task",
        Mock(return_value=(lifecycle, ExitCode.SUCCESS)),
    )
    monkeypatch.setattr(run_connector_module, "FORCE_EXIT_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(run_connector_module.os, "_exit", force_exit)

    assert (
        run_connector_module.run_connector_once("runtime.example:9092", "token", True)
        == 0
    )

    force_exit.assert_called_once_with(0)


def test_cold_connector_keeps_flower_exit_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cold CLI should keep Flower exit behavior around the task lifecycle."""
    lifecycle = Mock()
    run_task = Mock(return_value=(lifecycle, ExitCode.TASK_PROC_EXCEPTION))
    flwr_exit = Mock()
    monkeypatch.setattr(run_connector_module, "_run_connector_task", run_task)
    monkeypatch.setattr(run_connector_module, "flwr_exit", flwr_exit)

    run_connector_module.run_connector(
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
        event_type=EventType.FLWR_CONNECTOR_RUN_LEAVE,
    )
