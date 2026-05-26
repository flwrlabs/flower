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

from flwr.supercore.interceptors import (
    RuntimeVersionClientInterceptor,
    SuperExecAuthClientInterceptor,
)
from flwr.supercore.superexec.executor import LaunchResult, LaunchResultStatus

from . import run_superexec as run_superexec_module


def _run_superexec_one_launch(
    monkeypatch: pytest.MonkeyPatch, launch_result: LaunchResult | None
) -> tuple[Mock, Mock, Mock]:
    """Run one SuperExec launch loop and stop at the loop sleep."""
    channel = Mock()
    task = Mock()
    task.task_id = 123
    stub = Mock()
    stub.PullPendingTasks.return_value = Mock(tasks=[task])
    stub.ClaimTask.return_value = Mock(token="token-123")
    plugin = Mock()
    plugin.select_task.return_value = task
    plugin.launch_task.return_value = launch_result
    log = Mock()

    monkeypatch.setattr(
        run_superexec_module, "create_channel", Mock(return_value=channel)
    )
    monkeypatch.setattr(run_superexec_module, "register_signal_handlers", Mock())
    monkeypatch.setattr(run_superexec_module, "wrap_stub", Mock())
    monkeypatch.setattr(run_superexec_module, "get_executor", Mock())
    monkeypatch.setattr(run_superexec_module, "log", log)
    monkeypatch.setattr(
        "flwr.supercore.superexec.run_superexec.time.sleep",
        Mock(side_effect=KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        run_superexec_module.run_superexec(
            plugin_class=Mock(return_value=plugin),
            stub_class=Mock(return_value=stub),
            appio_api_address="127.0.0.1:9091",
            insecure=True,
        )

    return log, plugin, stub


@pytest.mark.parametrize(
    ("superexec_auth_secret", "expected_interceptor_types"),
    [
        (None, (RuntimeVersionClientInterceptor,)),
        (
            b"superexec-secret",
            (RuntimeVersionClientInterceptor, SuperExecAuthClientInterceptor),
        ),
    ],
)
def test_run_superexec_adds_runtime_version_interceptor(
    monkeypatch: pytest.MonkeyPatch,
    superexec_auth_secret: bytes | None,
    expected_interceptor_types: tuple[type[object], ...],
) -> None:
    """SuperExec should attach runtime version metadata to AppIO calls."""
    channel = Mock()
    stub = Mock()
    stub.PullPendingTasks.side_effect = KeyboardInterrupt()
    captured: dict[str, Any] = {}

    def _create_channel(**kwargs: Any) -> Mock:
        captured.update(kwargs)
        return channel

    monkeypatch.setattr(run_superexec_module, "create_channel", _create_channel)
    monkeypatch.setattr(run_superexec_module, "register_signal_handlers", Mock())
    monkeypatch.setattr(run_superexec_module, "wrap_stub", Mock())

    with pytest.raises(KeyboardInterrupt):
        run_superexec_module.run_superexec(
            plugin_class=Mock(),
            stub_class=Mock(return_value=stub),
            appio_api_address="127.0.0.1:9091",
            insecure=True,
            superexec_auth_secret=superexec_auth_secret,
        )

    assert tuple(type(interceptor) for interceptor in captured["interceptors"]) == (
        expected_interceptor_types
    )


def test_run_superexec_preserves_accepted_launch_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SuperExec should launch and continue quietly when launch is accepted."""
    log, plugin, stub = _run_superexec_one_launch(monkeypatch, LaunchResult.accepted())

    stub.ClaimTask.assert_called_once()
    plugin.launch_task.assert_called_once()
    log.assert_not_called()


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
    log, plugin, stub = _run_superexec_one_launch(monkeypatch, launch_result)

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
    log, plugin, stub = _run_superexec_one_launch(monkeypatch, None)

    stub.ClaimTask.assert_called_once()
    plugin.launch_task.assert_called_once()
    log.assert_not_called()


def test_handle_launch_result_handles_all_statuses() -> None:
    """All defined launch result statuses should be handled explicitly."""
    task = Mock()
    task.task_id = 123

    for status in LaunchResultStatus:
        run_superexec_module._handle_launch_result(  # pylint: disable=protected-access
            LaunchResult(status=status), task
        )
