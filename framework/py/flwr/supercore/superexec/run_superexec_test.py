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
"""Tests for SuperExec runtime setup."""


from collections.abc import Sequence
from logging import ERROR, WARNING
from typing import Any
from unittest.mock import Mock

import httpx
import pytest

from flwr.proto.runtime_pb2 import PullAndClaimTaskResponse  # pylint: disable=E0611
from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.supercore.constant import ExecutorType, TaskType
from flwr.supercore.interceptors import (
    RuntimeVersionHttpInterceptor,
    SuperExecAuthHttpInterceptor,
)
from flwr.supercore.superexec.executor import LaunchResult, LaunchResultStatus
from flwr.supercore.superexec.plugin import (
    AutoExecPlugin,
    ClientAppExecPlugin,
    ServerAppExecPlugin,
)

from . import run_superexec as run_superexec_module


def _run_superexec_one_launch(
    monkeypatch: pytest.MonkeyPatch,
    launch_result: LaunchResult,
) -> tuple[Mock, Mock, Mock, Mock]:
    """Run one SuperExec launch loop and stop at the next acquisition."""
    task = Mock()
    task.task_id = 123
    client = Mock()
    client.PullPendingTasks.side_effect = [Mock(tasks=[task]), KeyboardInterrupt()]
    client.ClaimTask.return_value = Mock(token="token-123")
    client_class = Mock()
    client_class.from_server_address.return_value = client
    plugin = Mock()
    plugin.select_task.return_value = task
    plugin.launch_task.return_value = launch_result
    log = Mock()

    monkeypatch.setattr(run_superexec_module, "register_signal_handlers", Mock())
    monkeypatch.setattr(run_superexec_module, "get_executor", Mock())
    monkeypatch.setattr(run_superexec_module, "log", log)
    sleep_mock = Mock()
    monkeypatch.setattr("flwr.supercore.superexec.run_superexec.time.sleep", sleep_mock)

    with pytest.raises(KeyboardInterrupt):
        run_superexec_module.run_superexec(
            plugin_class=Mock(return_value=plugin),
            client_class=client_class,
            runtime_api_address="127.0.0.1:9091",
            insecure=True,
        )

    return log, plugin, client, sleep_mock


@pytest.mark.parametrize(
    ("plugin_class", "task_type"),
    [
        (AutoExecPlugin, TaskType.MODEL),
        (ServerAppExecPlugin, TaskType.SERVER_APP),
        (ClientAppExecPlugin, TaskType.CLIENT_APP),
    ],
)
def test_builtin_subprocess_uses_one_acquisition_call(
    monkeypatch: pytest.MonkeyPatch,
    plugin_class: type[AutoExecPlugin | ServerAppExecPlugin | ClientAppExecPlugin],
    task_type: TaskType,
) -> None:
    """Built-in subprocess execution gates capacity before one claim request."""
    task = Task(task_id=123, type=task_type)
    client = Mock()
    client.PullAndClaimTask.side_effect = [
        PullAndClaimTaskResponse(task=task, token="task-token"),
        KeyboardInterrupt(),
    ]
    client_class = Mock()
    client_class.from_server_address.return_value = client
    executor = Mock()
    executor.launch.return_value = LaunchResult.accepted()
    get_executor = Mock(return_value=executor)
    order = Mock()
    order.attach_mock(executor.wait_for_capacity, "capacity")
    order.attach_mock(client.PullAndClaimTask, "acquire")
    monkeypatch.setattr(run_superexec_module, "get_executor", get_executor)
    monkeypatch.setattr(run_superexec_module, "register_signal_handlers", Mock())
    sleep = Mock()
    monkeypatch.setattr("flwr.supercore.superexec.run_superexec.time.sleep", sleep)

    with pytest.raises(KeyboardInterrupt):
        run_superexec_module.run_superexec(
            plugin_class=plugin_class,
            client_class=client_class,
            runtime_api_address="127.0.0.1:9091",
            insecure=True,
        )

    assert [call[0] for call in order.mock_calls[:2]] == ["capacity", "acquire"]
    assert set(client.PullAndClaimTask.call_args.args[0].supported_task_types) == set(
        plugin_class.supported_task_types
    )
    assert client.PullAndClaimTask.call_args.args[0].wait_timeout_ms == 5_000
    client.PullPendingTasks.assert_not_called()
    client.ClaimTask.assert_not_called()
    assert executor.launch.call_args.args[0].task_id == task.task_id
    assert executor.launch.call_args.args[0].token == "task-token"
    sleep.assert_not_called()


def test_builtin_subprocess_does_not_launch_for_empty_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty combined response leaves the executor idle."""
    client = Mock()
    client.PullAndClaimTask.side_effect = [
        PullAndClaimTaskResponse(),
        KeyboardInterrupt(),
    ]
    client_class = Mock()
    client_class.from_server_address.return_value = client
    executor = Mock()
    monkeypatch.setattr(
        run_superexec_module, "get_executor", Mock(return_value=executor)
    )
    monkeypatch.setattr(run_superexec_module, "register_signal_handlers", Mock())
    sleep = Mock()
    monkeypatch.setattr("flwr.supercore.superexec.run_superexec.time.sleep", sleep)

    with pytest.raises(KeyboardInterrupt):
        run_superexec_module.run_superexec(
            plugin_class=AutoExecPlugin,
            client_class=client_class,
            runtime_api_address="127.0.0.1:9091",
            insecure=True,
        )

    assert client.PullAndClaimTask.call_count == 2
    sleep.assert_not_called()
    executor.launch.assert_not_called()


@pytest.mark.parametrize(("interval", "expected"), [(None, 1.0), ("0.25", 0.25)])
def test_combined_acquisition_backs_off_after_connection_failure(
    monkeypatch: pytest.MonkeyPatch, interval: str | None, expected: float
) -> None:
    """A connection failure before sending a claim can be retried safely."""
    if interval is None:
        monkeypatch.delenv("FLWR_SUPEREXEC_TASK_POLL_INTERVAL", raising=False)
    else:
        monkeypatch.setenv("FLWR_SUPEREXEC_TASK_POLL_INTERVAL", interval)
    client = Mock()
    client.PullAndClaimTask.side_effect = [
        httpx.ConnectError("unavailable"),
        KeyboardInterrupt(),
    ]
    client_class = Mock()
    client_class.from_server_address.return_value = client
    executor = Mock()
    monkeypatch.setattr(
        run_superexec_module, "get_executor", Mock(return_value=executor)
    )
    monkeypatch.setattr(run_superexec_module, "register_signal_handlers", Mock())
    sleep = Mock()
    monkeypatch.setattr("flwr.supercore.superexec.run_superexec.time.sleep", sleep)

    with pytest.raises(KeyboardInterrupt):
        run_superexec_module.run_superexec(
            plugin_class=AutoExecPlugin,
            client_class=client_class,
            runtime_api_address="127.0.0.1:9091",
            insecure=True,
        )

    sleep.assert_called_once_with(expected)
    assert client.PullAndClaimTask.call_count == 2


def test_combined_acquisition_stops_after_lost_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ambiguous claim waits for lease expiry before another acquisition."""
    client = Mock()
    client.PullAndClaimTask.side_effect = [
        httpx.ReadError("response lost"),
        KeyboardInterrupt(),
    ]
    client_class = Mock()
    client_class.from_server_address.return_value = client
    executor = Mock()
    monkeypatch.setattr(
        run_superexec_module, "get_executor", Mock(return_value=executor)
    )
    monkeypatch.setattr(run_superexec_module, "register_signal_handlers", Mock())
    sleep = Mock()
    monkeypatch.setattr("flwr.supercore.superexec.run_superexec.time.sleep", sleep)

    with pytest.raises(KeyboardInterrupt):
        run_superexec_module.run_superexec(
            plugin_class=AutoExecPlugin,
            client_class=client_class,
            runtime_api_address="127.0.0.1:9091",
            insecure=True,
        )

    assert client.PullAndClaimTask.call_count == 2
    assert sleep.call_count == 12
    assert all(call.args == (5.0,) for call in sleep.call_args_list)
    assert executor.reconcile.call_count == 14
    client.close.assert_called_once()


def test_kubernetes_keeps_task_specific_capacity_before_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kubernetes waits with the selected task type before claiming."""
    task = Task(task_id=123, type=TaskType.MODEL)
    client = Mock()
    client.PullPendingTasks.side_effect = [Mock(tasks=[task]), KeyboardInterrupt()]
    client.ClaimTask.return_value = Mock(token="task-token")
    client_class = Mock()
    client_class.from_server_address.return_value = client
    executor = Mock()
    executor.launch.return_value = LaunchResult.accepted()
    order = Mock()
    order.attach_mock(client.PullPendingTasks, "pull")
    order.attach_mock(executor.wait_for_capacity, "capacity")
    order.attach_mock(client.ClaimTask, "claim")
    monkeypatch.setattr(
        run_superexec_module, "get_executor", Mock(return_value=executor)
    )
    monkeypatch.setattr(run_superexec_module, "register_signal_handlers", Mock())
    sleep = Mock()
    monkeypatch.setattr("flwr.supercore.superexec.run_superexec.time.sleep", sleep)

    with pytest.raises(KeyboardInterrupt):
        run_superexec_module.run_superexec(
            plugin_class=AutoExecPlugin,
            client_class=client_class,
            runtime_api_address="127.0.0.1:9091",
            insecure=True,
            executor_type=ExecutorType.KUBERNETES,
        )

    assert [call[0] for call in order.mock_calls[:3]] == ["pull", "capacity", "claim"]
    assert client.PullPendingTasks.call_args.kwargs["request"].wait_timeout_ms == 5_000
    assert executor.wait_for_capacity.call_args.kwargs["task_type"] == TaskType.MODEL
    client.PullAndClaimTask.assert_not_called()
    sleep.assert_not_called()


def test_kubernetes_reconciles_without_extra_sleep_after_empty_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty legacy long poll immediately returns to reconciliation."""
    client = Mock()
    client.PullPendingTasks.side_effect = [Mock(tasks=[]), KeyboardInterrupt()]
    client_class = Mock()
    client_class.from_server_address.return_value = client
    executor = Mock()
    monkeypatch.setattr(
        run_superexec_module, "get_executor", Mock(return_value=executor)
    )
    monkeypatch.setattr(run_superexec_module, "register_signal_handlers", Mock())
    sleep = Mock()
    monkeypatch.setattr("flwr.supercore.superexec.run_superexec.time.sleep", sleep)

    with pytest.raises(KeyboardInterrupt):
        run_superexec_module.run_superexec(
            plugin_class=AutoExecPlugin,
            client_class=client_class,
            runtime_api_address="127.0.0.1:9091",
            insecure=True,
            executor_type=ExecutorType.KUBERNETES,
        )

    assert client.PullPendingTasks.call_count == 2
    assert executor.reconcile.call_count == 2
    sleep.assert_not_called()
    client.ClaimTask.assert_not_called()


def test_custom_plugin_keeps_its_selection_logic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plugin override must still receive all pending candidates."""

    class SelectLastPlugin(AutoExecPlugin):
        """Choose the last pending task."""

        def select_task(self, candidate_tasks: Sequence[Task]) -> Task | None:
            """Select the last candidate."""
            return candidate_tasks[-1] if candidate_tasks else None

    client = Mock()
    client.PullPendingTasks.side_effect = [
        Mock(
            tasks=[
                Task(task_id=1, type=TaskType.MODEL),
                Task(task_id=2, type=TaskType.MODEL),
            ]
        ),
        KeyboardInterrupt(),
    ]
    client.ClaimTask.return_value = Mock(token="task-token")
    client_class = Mock()
    client_class.from_server_address.return_value = client
    executor = Mock()
    executor.launch.return_value = LaunchResult.accepted()
    monkeypatch.setattr(
        run_superexec_module, "get_executor", Mock(return_value=executor)
    )
    monkeypatch.setattr(run_superexec_module, "register_signal_handlers", Mock())
    sleep = Mock()
    monkeypatch.setattr("flwr.supercore.superexec.run_superexec.time.sleep", sleep)

    with pytest.raises(KeyboardInterrupt):
        run_superexec_module.run_superexec(
            plugin_class=SelectLastPlugin,
            client_class=client_class,
            runtime_api_address="127.0.0.1:9091",
            insecure=True,
        )

    assert client.ClaimTask.call_args.args[0].task_id == 2
    client.PullAndClaimTask.assert_not_called()
    sleep.assert_not_called()


@pytest.mark.parametrize(
    ("superexec_auth_secret", "expected_interceptor_types"),
    [
        (None, (RuntimeVersionHttpInterceptor,)),
        (
            b"superexec-secret",
            (RuntimeVersionHttpInterceptor, SuperExecAuthHttpInterceptor),
        ),
    ],
)
def test_run_superexec_adds_runtime_version_interceptor(
    monkeypatch: pytest.MonkeyPatch,
    superexec_auth_secret: bytes | None,
    expected_interceptor_types: tuple[type[object], ...],
) -> None:
    """SuperExec should attach runtime version metadata to Runtime API calls."""
    client = Mock()
    client.PullPendingTasks.side_effect = KeyboardInterrupt()
    client_class = Mock()
    captured: dict[str, Any] = {}

    def _from_server_address(**kwargs: Any) -> Mock:
        captured.update(kwargs)
        return client

    client_class.from_server_address.side_effect = _from_server_address
    monkeypatch.setattr(run_superexec_module, "register_signal_handlers", Mock())

    with pytest.raises(KeyboardInterrupt):
        run_superexec_module.run_superexec(
            plugin_class=Mock(),
            client_class=client_class,
            runtime_api_address="127.0.0.1:9091",
            insecure=True,
            superexec_auth_secret=superexec_auth_secret,
        )

    assert tuple(type(interceptor) for interceptor in captured["interceptors"]) == (
        expected_interceptor_types
    )
    if superexec_auth_secret:
        auth_interceptor = captured["interceptors"][1]
        assert (
            "/flwr.proto.Runtime/PullAndClaimTask"
            in auth_interceptor._protected_methods  # pylint: disable=protected-access
        )


@pytest.mark.parametrize(
    ("insecure", "root_certificates_path"), [(True, None), (False, "runtime-ca.pem")]
)
def test_run_superexec_passes_executor_config_to_factory(
    monkeypatch: pytest.MonkeyPatch,
    insecure: bool,
    root_certificates_path: str | None,
) -> None:
    """SuperExec should pass executor config and Runtime transport to the factory."""
    client = Mock()
    client.PullPendingTasks.side_effect = KeyboardInterrupt()
    client_class = Mock()
    client_class.from_server_address.return_value = client
    executor_config: dict[str, object] = {
        "namespace": "flower-system",
        "image": "taskexecutor:dev",
    }
    get_executor = Mock(return_value=Mock())

    monkeypatch.setattr(run_superexec_module, "register_signal_handlers", Mock())
    monkeypatch.setattr(run_superexec_module, "get_executor", get_executor)
    monkeypatch.setattr(
        run_superexec_module, "validate_and_resolve_root_certificates", Mock()
    )

    with pytest.raises(KeyboardInterrupt):
        run_superexec_module.run_superexec(
            plugin_class=Mock(),
            client_class=client_class,
            runtime_api_address="127.0.0.1:9091",
            insecure=insecure,
            root_certificates_path=root_certificates_path,
            executor_type=ExecutorType.KUBERNETES,
            executor_config=executor_config,
        )

    get_executor.assert_called_once_with(
        ExecutorType.KUBERNETES,
        executor_config=executor_config,
        insecure=insecure,
        root_certificates_path=root_certificates_path,
    )
    get_executor.return_value.reconcile.assert_called_once_with()
    get_executor.return_value.close.assert_called_once_with()


def test_run_superexec_closes_executor_when_runtime_client_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warm Pods are cleaned up when startup fails before handlers are installed."""
    executor = Mock()
    client_class = Mock()
    client_class.from_server_address.side_effect = RuntimeError("Runtime unavailable")
    monkeypatch.setattr(
        run_superexec_module, "get_executor", Mock(return_value=executor)
    )

    with pytest.raises(RuntimeError, match="Runtime unavailable"):
        run_superexec_module.run_superexec(
            plugin_class=Mock(),
            client_class=client_class,
            runtime_api_address="127.0.0.1:9091",
            insecure=True,
        )

    executor.close.assert_called_once_with()


def test_run_superexec_preserves_accepted_launch_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SuperExec should launch and continue quietly when launch is accepted."""
    log, plugin, stub, sleep_mock = _run_superexec_one_launch(
        monkeypatch, LaunchResult.accepted()
    )

    stub.ClaimTask.assert_called_once()
    plugin.launch_task.assert_called_once()
    log.assert_not_called()
    sleep_mock.assert_not_called()


@pytest.mark.parametrize(
    ("launch_result", "expected_level", "expected_message"),
    [
        (
            LaunchResult.capacity_rejected("namespace quota exceeded"),
            WARNING,
            "Executor rejected launch",
        ),
        (
            LaunchResult.failed("invalid execution spec"),
            ERROR,
            "Executor failed to launch",
        ),
        (
            LaunchResult.unknown("create request timed out"),
            WARNING,
            "Executor launch outcome is unknown",
        ),
    ],
)
def test_run_superexec_logs_non_accepted_launch_result(
    monkeypatch: pytest.MonkeyPatch,
    launch_result: LaunchResult,
    expected_level: int,
    expected_message: str,
) -> None:
    """SuperExec should log non-accepted launch results and keep loop behavior."""
    log, plugin, stub, _ = _run_superexec_one_launch(monkeypatch, launch_result)

    stub.ClaimTask.assert_called_once()
    plugin.launch_task.assert_called_once()
    log.assert_called_once()
    assert log.call_args.args[0] == expected_level
    assert expected_message in log.call_args.args[1]
    assert log.call_args.args[2] == 123


@pytest.mark.parametrize(
    "value", ["", "0", "0.009", "-1", "60.001", "1e20", "nan", "inf", "not-a-number"]
)
def test_run_superexec_rejects_invalid_task_poll_interval(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """SuperExec should reject invalid task polling intervals."""
    monkeypatch.setenv("FLWR_SUPEREXEC_TASK_POLL_INTERVAL", value)

    with pytest.raises(ValueError, match="FLWR_SUPEREXEC_TASK_POLL_INTERVAL"):
        run_superexec_module.run_superexec(
            plugin_class=Mock(),
            client_class=Mock(),
            runtime_api_address="127.0.0.1:9091",
            insecure=True,
        )


@pytest.mark.parametrize("value", ["0.01", "60"])
def test_run_superexec_accepts_task_poll_interval_bounds(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """SuperExec should accept the configured polling interval bounds."""
    monkeypatch.setenv("FLWR_SUPEREXEC_TASK_POLL_INTERVAL", value)

    # pylint: disable-next=protected-access
    get_task_poll_interval = run_superexec_module._get_task_poll_interval
    assert get_task_poll_interval() == float(value)


def test_handle_launch_result_handles_all_statuses() -> None:
    """All defined launch result statuses should be handled explicitly."""
    task = Mock()
    task.task_id = 123

    for status in LaunchResultStatus:
        run_superexec_module._handle_launch_result(  # pylint: disable=protected-access
            LaunchResult(status=status), task
        )
