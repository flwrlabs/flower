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
"""Tests for SuperExec base ephemeral plugin behavior."""


from unittest.mock import MagicMock, Mock, patch

import pytest

from flwr.supercore.constant import TaskType
from flwr.supercore.exit import ExitCode

from .base_ephemeral_exec_plugin import BaseEphemeralExecPlugin


def _get_task(*, task_id: int = 1, task_type: str = TaskType.SERVER_APP) -> Mock:
    """Return a minimal dummy task-like object."""
    task = Mock()
    task.task_id = task_id
    task.type = task_type
    return task


class _EphemeralExecPlugin(BaseEphemeralExecPlugin):
    command = "flwr-serverapp"
    runtime_api_address_arg = "--runtime-api-address"


def _get_ephemeral_plugin() -> _EphemeralExecPlugin:
    return _EphemeralExecPlugin(
        runtime_api_address="127.0.0.1:9091",
        insecure=True,
        root_certificates_path=None,
    )


def test_select_run_id_returns_none_when_no_candidates() -> None:
    """The plugin should skip execution when no runs are available."""
    plugin = _get_ephemeral_plugin()
    assert plugin.select_run_id([]) is None


def test_select_run_id_returns_first_candidate() -> None:
    """The plugin should always choose the first candidate run ID."""
    plugin = _get_ephemeral_plugin()
    assert plugin.select_run_id([7, 9, 11]) == 7


def test_launch_task_runs_expected_command_and_exits() -> None:
    """Launch should invoke the app with token and parent PID, then exit."""
    plugin = _get_ephemeral_plugin()

    with (
        patch(
            "flwr.supercore.superexec.plugin.base_ephemeral_exec_plugin.os.getpid",
            return_value=1234,
        ),
        patch(
            "flwr.supercore.superexec.plugin.base_ephemeral_exec_plugin.subprocess.Popen"
        ) as popen,
        patch(
            "flwr.supercore.superexec.plugin.base_ephemeral_exec_plugin.flwr_exit"
        ) as flwr_exit,
    ):
        plugin.launch_task(token="token-123", task=_get_task(task_id=5))

    popen.assert_called_once_with(
        [
            "flwr-serverapp",
            "--insecure",
            "--runtime-api-address",
            "127.0.0.1:9091",
            "--token",
            "token-123",
            "--parent-pid",
            "1234",
        ],
    )
    flwr_exit.assert_called_once_with(
        ExitCode.SUCCESS,
        "App process finished, exiting SuperExec.",
    )


def test_launch_task_calls_cleanup_before_launch() -> None:
    """Launch should invoke cleanup_before_launch before running the subprocess."""
    # Prepare
    call_log: list[str] = []
    plugin = _get_ephemeral_plugin()
    plugin.cleanup_before_launch = lambda: call_log.append("cleanup")

    def popen(*_args: object, **_kwargs: object) -> MagicMock:
        call_log.append("subprocess")
        return MagicMock()

    # Execute
    with (
        patch(
            "flwr.supercore.superexec.plugin.base_ephemeral_exec_plugin.subprocess.Popen",
            side_effect=popen,
        ),
        patch("flwr.supercore.superexec.plugin.base_ephemeral_exec_plugin.flwr_exit"),
    ):
        plugin.launch_task(token="token-abc", task=_get_task(task_id=1))

    # Assert
    assert call_log == ["cleanup", "subprocess"]


def test_launch_task_forwards_timing_context_for_agent_tasks() -> None:
    """Ephemeral Agent launches should preserve timing correlation metadata."""
    plugin = _get_ephemeral_plugin()
    task = _get_task(task_id=5, task_type=TaskType.AGENT_APP)
    task.run_id = 7

    with (
        patch(
            "flwr.supercore.superexec.plugin.base_ephemeral_exec_plugin."
            "is_runtime_timing_logging_enabled",
            return_value=True,
        ),
        patch(
            "flwr.supercore.superexec.plugin.base_ephemeral_exec_plugin."
            "emit_runtime_timing"
        ) as emit_runtime_timing,
        patch(
            "flwr.supercore.superexec.plugin.base_ephemeral_exec_plugin.subprocess.Popen"
        ) as popen,
        patch("flwr.supercore.superexec.plugin.base_ephemeral_exec_plugin.flwr_exit"),
    ):

        def assert_spawn_started_before_popen(
            *_args: object, **_kwargs: object
        ) -> Mock:
            assert [call.args[0] for call in emit_runtime_timing.call_args_list] == [
                "runtime.executor.child.spawn.started"
            ]
            return MagicMock()

        popen.side_effect = assert_spawn_started_before_popen
        plugin.launch_task(token="token-123", task=task)

    assert popen.call_args.args[0][-4:] == [
        "--runtime-timing-run-id",
        "7",
        "--runtime-timing-task-id",
        "5",
    ]
    assert [call.args[0] for call in emit_runtime_timing.call_args_list] == [
        "runtime.executor.child.spawn.started",
        "runtime.executor.submission.accepted",
        "runtime.task.dispatch.accepted",
    ]


def test_launch_task_emits_no_acceptance_markers_when_spawn_fails() -> None:
    """A failed child launch must not look like an accepted dispatch."""
    plugin = _get_ephemeral_plugin()
    task = _get_task(task_id=5, task_type=TaskType.AGENT_APP)
    task.run_id = 7

    with (
        patch(
            "flwr.supercore.superexec.plugin.base_ephemeral_exec_plugin."
            "is_runtime_timing_logging_enabled",
            return_value=True,
        ),
        patch(
            "flwr.supercore.superexec.plugin.base_ephemeral_exec_plugin."
            "emit_runtime_timing"
        ) as emit_runtime_timing,
        patch(
            "flwr.supercore.superexec.plugin.base_ephemeral_exec_plugin.subprocess.Popen",
            side_effect=FileNotFoundError,
        ),
    ):
        with pytest.raises(FileNotFoundError):
            plugin.launch_task(token="token-123", task=task)

    assert [call.args[0] for call in emit_runtime_timing.call_args_list] == [
        "runtime.executor.child.spawn.started"
    ]
