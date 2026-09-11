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


import os
from unittest.mock import ANY, Mock, patch

import pytest

from flwr.supercore.constant import TaskType
from flwr.supercore.exit import ExitCode

from .base_ephemeral_exec_plugin import BaseEphemeralExecPlugin

_PROVIDER_ENV = {
    "BRAVE_API_KEY": "brave-key",
    "EXA_API_KEY": "exa-key",
    "FLWR_MODEL_API_KEY": "model-key",
    "TAVILY_API_KEY": "tavily-key",
}


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
            "--runtime-api-address",
            "127.0.0.1:9091",
            "--token",
            "token-123",
            "--parent-pid",
            "1234",
        ],
        check=False,
        env=ANY,
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


@pytest.mark.parametrize(
    "task_type",
    [TaskType.AGENT_APP, TaskType.SERVER_APP, TaskType.SIMULATION],
    ids=["agentapp", "serverapp", "simulation"],
)
def test_launch_task_removes_provider_keys_for_fab_backed_tasks(
    task_type: TaskType,
) -> None:
    """Launch should remove provider keys from FAB-backed task env."""
    plugin = _get_ephemeral_plugin()
    env = {
        **_PROVIDER_ENV,
        "PATH": "/usr/bin",
        "PYTHONPATH": "/path/to/python",
        "UNRELATED_API_KEY": "keep-me",
    }

    with (
        patch.dict(os.environ, env, clear=True),
        patch(
            "flwr.supercore.superexec.plugin.base_ephemeral_exec_plugin.subprocess.run"
        ) as run,
        patch("flwr.supercore.superexec.plugin.base_ephemeral_exec_plugin.flwr_exit"),
    ):
        plugin.launch_task(token="token", task=_get_task(task_type=task_type))

    child_env = run.call_args.kwargs["env"]
    for env_var in _PROVIDER_ENV:
        assert env_var not in child_env
    assert child_env["PATH"] == "/usr/bin"
    assert child_env["PYTHONPATH"] == "/path/to/python"
    assert child_env["UNRELATED_API_KEY"] == "keep-me"


@pytest.mark.parametrize(
    "task_type",
    [TaskType.CONNECTOR, TaskType.MODEL],
    ids=["connector", "model"],
)
def test_launch_task_keeps_provider_keys_for_flower_controlled_tasks(
    task_type: TaskType,
) -> None:
    """Launch should keep provider keys for Flower-controlled task env."""
    plugin = _get_ephemeral_plugin()
    env = {**_PROVIDER_ENV, "PATH": "/usr/bin"}

    with (
        patch.dict(os.environ, env, clear=True),
        patch(
            "flwr.supercore.superexec.plugin.base_ephemeral_exec_plugin.subprocess.run"
        ) as run,
        patch("flwr.supercore.superexec.plugin.base_ephemeral_exec_plugin.flwr_exit"),
    ):
        plugin.launch_task(token="token", task=_get_task(task_type=task_type))

    child_env = run.call_args.kwargs["env"]
    for env_var, value in _PROVIDER_ENV.items():
        assert child_env[env_var] == value
