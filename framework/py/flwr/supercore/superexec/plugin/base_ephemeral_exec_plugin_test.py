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


from unittest.mock import Mock, patch

import pytest

from flwr.common.exit import ExitCode
from flwr.common.typing import Run
from flwr.supercore.constant import TaskType

from .base_ephemeral_exec_plugin import BaseEphemeralExecPlugin
from .serverapp_ephemeral_exec_plugin import ServerAppEphemeralExecPlugin


def _get_run(_: int) -> Run:
    """Return a minimal dummy run."""
    return Run.create_empty(run_id=1)


def _get_task(*, task_id: int = 1, task_type: str = TaskType.SERVER_APP) -> Mock:
    """Return a minimal dummy task-like object."""
    task = Mock()
    task.task_id = task_id
    task.type = task_type
    return task


class _EphemeralExecPlugin(BaseEphemeralExecPlugin):
    command = "flwr-serverapp"
    appio_api_address_arg = "--serverappio-api-address"


def _get_ephemeral_plugin() -> _EphemeralExecPlugin:
    return _EphemeralExecPlugin(
        appio_api_address="127.0.0.1:9091",
        get_run=_get_run,
        insecure=True,
        root_certificates_path=None,
    )


def _get_serverapp_ephemeral_plugin() -> ServerAppEphemeralExecPlugin:
    return ServerAppEphemeralExecPlugin(
        appio_api_address="127.0.0.1:9092",
        get_run=_get_run,
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


def test_serverapp_ephemeral_select_task_skips_unsupported_tasks() -> None:
    """ServerApp ephemeral plugin should select the first supported task."""
    plugin = _get_serverapp_ephemeral_plugin()
    unsupported_task = _get_task(task_id=1, task_type=TaskType.CLIENT_APP)
    agentapp_task = _get_task(task_id=2, task_type=TaskType.AGENT_APP)

    assert plugin.select_task([unsupported_task, agentapp_task]) is agentapp_task
    assert plugin.select_task([unsupported_task]) is None


def test_launch_task_runs_expected_command_and_exits() -> None:
    """Launch should invoke the app with token and parent PID, then exit."""
    plugin = _get_ephemeral_plugin()

    with (
        patch(
            "flwr.supercore.superexec.plugin.base_ephemeral_exec_plugin.os.getpid",
            return_value=1234,
        ),
        patch(
            "flwr.supercore.superexec.plugin.base_ephemeral_exec_plugin.subprocess.run"
        ) as run,
        patch(
            "flwr.supercore.superexec.plugin.base_ephemeral_exec_plugin.flwr_exit"
        ) as flwr_exit,
    ):
        plugin.launch_task(token="token-123", task=_get_task(task_id=5))

    run.assert_called_once_with(
        [
            "flwr-serverapp",
            "--insecure",
            "--serverappio-api-address",
            "127.0.0.1:9091",
            "--token",
            "token-123",
            "--parent-pid",
            "1234",
        ],
        check=False,
    )
    flwr_exit.assert_called_once_with(
        ExitCode.SUCCESS,
        "App process finished, exiting SuperExec.",
    )


@pytest.mark.parametrize(
    ("task_type", "command"),
    [
        (TaskType.SERVER_APP, "flwr-serverapp"),
        (TaskType.SIMULATION, "flwr-simulation"),
        (TaskType.AGENT_APP, "flwr-agentapp"),
        (TaskType.MODEL, "flwr-model"),
    ],
)
def test_serverapp_ephemeral_launches_supported_superlink_task_commands(
    task_type: TaskType, command: str
) -> None:
    """ServerApp ephemeral plugin should use fixed SuperLink task commands."""
    plugin = _get_serverapp_ephemeral_plugin()

    with (
        patch(
            "flwr.supercore.superexec.plugin.base_ephemeral_exec_plugin.os.getpid",
            return_value=1234,
        ),
        patch(
            "flwr.supercore.superexec.plugin.base_ephemeral_exec_plugin.subprocess.run"
        ) as run,
        patch(
            "flwr.supercore.superexec.plugin.base_ephemeral_exec_plugin.flwr_exit"
        ) as flwr_exit,
    ):
        plugin.launch_task(
            token="token-123",
            task=_get_task(task_id=5, task_type=task_type),
        )

    run.assert_called_once_with(
        [
            command,
            "--insecure",
            "--serverappio-api-address",
            "127.0.0.1:9092",
            "--token",
            "token-123",
            "--parent-pid",
            "1234",
        ],
        check=False,
    )
    flwr_exit.assert_called_once_with(
        ExitCode.SUCCESS,
        "App process finished, exiting SuperExec.",
    )


def test_serverapp_ephemeral_launch_skips_unsupported_task_type() -> None:
    """ServerApp ephemeral plugin should not launch unsupported task types."""
    plugin = _get_serverapp_ephemeral_plugin()

    with (
        patch(
            "flwr.supercore.superexec.plugin.base_ephemeral_exec_plugin.subprocess.run"
        ) as run,
        patch(
            "flwr.supercore.superexec.plugin.base_ephemeral_exec_plugin.flwr_exit"
        ) as flwr_exit,
    ):
        plugin.launch_task(
            token="token-123",
            task=_get_task(task_id=5, task_type=TaskType.CLIENT_APP),
        )

    run.assert_not_called()
    flwr_exit.assert_not_called()


def test_launch_task_calls_cleanup_before_launch() -> None:
    """Launch should invoke cleanup_before_launch before running the subprocess."""
    # Prepare
    call_log: list[str] = []
    plugin = _get_ephemeral_plugin()
    plugin.cleanup_before_launch = lambda: call_log.append("cleanup")

    # Execute
    with (
        patch(
            "flwr.supercore.superexec.plugin.base_ephemeral_exec_plugin.subprocess.run",
            side_effect=lambda *_, **__: call_log.append("subprocess"),
        ),
        patch("flwr.supercore.superexec.plugin.base_ephemeral_exec_plugin.flwr_exit"),
    ):
        plugin.launch_task(token="token-abc", task=_get_task(task_id=1))

    # Assert
    assert call_log == ["cleanup", "subprocess"]
