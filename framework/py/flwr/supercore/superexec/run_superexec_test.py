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


from logging import ERROR, WARNING
from typing import Any
from unittest.mock import Mock

import pytest

from flwr.supercore.constant import ExecutorType
from flwr.supercore.interceptors import (
    RuntimeVersionHttpInterceptor,
    SuperExecAuthHttpInterceptor,
)
from flwr.supercore.superexec.executor import LaunchResult, LaunchResultStatus

from . import run_superexec as run_superexec_module


def _run_superexec_one_launch(
    monkeypatch: pytest.MonkeyPatch,
    launch_result: LaunchResult | None,
    task_poll_interval: str | None = None,
) -> tuple[Mock, Mock, Mock, Mock]:
    """Run one SuperExec launch loop and stop at the loop sleep."""
    if task_poll_interval is None:
        monkeypatch.delenv("FLWR_SUPEREXEC_TASK_POLL_INTERVAL", raising=False)
    else:
        monkeypatch.setenv("FLWR_SUPEREXEC_TASK_POLL_INTERVAL", task_poll_interval)

    task = Mock()
    task.task_id = 123
    client = Mock()
    client.PullPendingTasks.return_value = Mock(tasks=[task])
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
    sleep_mock = Mock(side_effect=KeyboardInterrupt())
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


def test_run_superexec_passes_executor_config_to_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SuperExec should pass selected executor config to the factory."""
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

    with pytest.raises(KeyboardInterrupt):
        run_superexec_module.run_superexec(
            plugin_class=Mock(),
            client_class=client_class,
            runtime_api_address="127.0.0.1:9091",
            insecure=True,
            executor_type=ExecutorType.KUBERNETES,
            executor_config=executor_config,
        )

    get_executor.assert_called_once_with(
        ExecutorType.KUBERNETES, executor_config=executor_config
    )


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
    sleep_mock.assert_called_once_with(0.1)


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


def test_run_superexec_continues_when_plugin_returns_no_launch_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SuperExec should not crash if a plugin returns no launch result."""
    log, plugin, stub, _ = _run_superexec_one_launch(monkeypatch, None)

    stub.ClaimTask.assert_called_once()
    plugin.launch_task.assert_called_once()
    log.assert_not_called()


def test_run_superexec_uses_configured_task_poll_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SuperExec should use the task polling interval from the environment."""
    _, _, _, sleep_mock = _run_superexec_one_launch(
        monkeypatch, LaunchResult.accepted(), task_poll_interval="0.25"
    )

    sleep_mock.assert_called_once_with(0.25)


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
