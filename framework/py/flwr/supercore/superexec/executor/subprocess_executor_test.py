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
"""Tests for SuperExec subprocess executor."""


import os
import subprocess
from typing import Any
from unittest.mock import ANY, Mock, patch

import pytest

from flwr.supercore.constant import TaskType

from .subprocess_executor import SubprocessExecutor
from .types import ExecutionSpec, LaunchResultStatus

_PROVIDER_ENV = {
    "BRAVE_API_KEY": "brave-key",
    "EXA_API_KEY": "exa-key",
    "FLWR_MODEL_API_KEY": "model-key",
    "TAVILY_API_KEY": "tavily-key",
}


def _execution_spec(**overrides: Any) -> ExecutionSpec:
    base: dict[str, Any] = {
        "task_type": TaskType.CLIENT_APP,
        "runtime_api_address": "127.0.0.1:9094",
        "token": "token",
        "insecure": True,
        "root_certificates_path": None,
        "runtime_dependency_install": False,
        "parent_pid": None,
        "suppress_output": False,
        "task_id": 1,
    }
    base.update(overrides)
    return ExecutionSpec(**base)


def test_launch_renders_insecure_clientapp_args() -> None:
    """Test subprocess executor renders insecure ClientApp args."""
    with patch.object(subprocess, "Popen") as popen_mock:
        result = SubprocessExecutor().launch(_execution_spec())

    popen_mock.assert_called_once_with(
        [
            "flwr-clientapp",
            "--runtime-api-address",
            "127.0.0.1:9094",
            "--token",
            "token",
            "--insecure",
        ],
        env=ANY,
    )
    assert result.status == LaunchResultStatus.ACCEPTED
    assert result.message is None


def test_launch_renders_root_certificates_args() -> None:
    """Test subprocess executor renders root certificates args."""
    with patch.object(subprocess, "Popen") as popen_mock:
        SubprocessExecutor().launch(
            _execution_spec(
                insecure=False,
                root_certificates_path="/path/to/root.pem",
            )
        )

    popen_mock.assert_called_once_with(
        [
            "flwr-clientapp",
            "--runtime-api-address",
            "127.0.0.1:9094",
            "--token",
            "token",
            "--root-certificates",
            "/path/to/root.pem",
        ],
        env=ANY,
    )


def test_launch_renders_runtime_dependency_install_flag() -> None:
    """Test subprocess executor renders runtime dependency installation flag."""
    with patch.object(subprocess, "Popen") as popen_mock:
        SubprocessExecutor().launch(_execution_spec(runtime_dependency_install=True))

    assert "--allow-runtime-dependency-installation" in popen_mock.call_args.args[0]


def test_launch_renders_parent_pid_flag() -> None:
    """Test subprocess executor renders subprocess parent PID flag."""
    with patch.object(subprocess, "Popen") as popen_mock:
        SubprocessExecutor().launch(_execution_spec(parent_pid=999))

    assert "--parent-pid" in popen_mock.call_args.args[0]
    assert "999" in popen_mock.call_args.args[0]


def test_launch_suppresses_output_when_requested() -> None:
    """Test subprocess executor suppresses output when requested."""
    with patch.object(subprocess, "Popen") as popen_mock:
        result = SubprocessExecutor().launch(_execution_spec(suppress_output=True))

    assert popen_mock.call_args.kwargs["env"] is not None
    assert popen_mock.call_args.kwargs["stdout"] == subprocess.DEVNULL
    assert popen_mock.call_args.kwargs["stderr"] == subprocess.DEVNULL
    assert result.status == LaunchResultStatus.ACCEPTED


@pytest.mark.parametrize(
    ("task_type", "command"),
    [
        (TaskType.SERVER_APP, "flwr-serverapp"),
        (TaskType.SIMULATION, "flwr-simulation"),
        (TaskType.AGENT_APP, "flwr-agentapp"),
        (TaskType.MODEL, "flwr-model"),
        (TaskType.CONNECTOR, "flwr-connector"),
    ],
    ids=["serverapp", "simulation", "agentapp", "model", "connector"],
)
def test_launch_renders_runtime_api_task_args(
    task_type: TaskType, command: str
) -> None:
    """Test subprocess executor renders Runtime API task args."""
    with patch.object(subprocess, "Popen") as popen_mock:
        SubprocessExecutor().launch(_execution_spec(task_type=task_type))

    popen_mock.assert_called_once_with(
        [
            command,
            "--runtime-api-address",
            "127.0.0.1:9094",
            "--token",
            "token",
            "--insecure",
        ],
        env=ANY,
    )


def test_launch_does_not_suppress_output_by_default() -> None:
    """Test subprocess executor leaves output inherited by default."""
    popen_mock = Mock()

    with patch.object(subprocess, "Popen", popen_mock):
        SubprocessExecutor().launch(_execution_spec())

    assert "stdout" not in popen_mock.call_args.kwargs
    assert "stderr" not in popen_mock.call_args.kwargs


@pytest.mark.parametrize(
    "task_type",
    [
        TaskType.AGENT_APP,
        TaskType.CLIENT_APP,
        TaskType.SERVER_APP,
        TaskType.SIMULATION,
    ],
    ids=["agentapp", "clientapp", "serverapp", "simulation"],
)
def test_launch_removes_provider_keys_for_fab_backed_tasks(
    task_type: TaskType,
) -> None:
    """Test subprocess executor removes provider keys from FAB-backed tasks."""
    env = {
        **_PROVIDER_ENV,
        "PATH": "/usr/bin",
        "PYTHONPATH": "/path/to/python",
        "UNRELATED_API_KEY": "keep-me",
    }

    with (
        patch.dict(os.environ, env, clear=True),
        patch.object(subprocess, "Popen") as popen_mock,
    ):
        SubprocessExecutor().launch(_execution_spec(task_type=task_type))

    child_env = popen_mock.call_args.kwargs["env"]
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
def test_launch_keeps_provider_keys_for_flower_controlled_tasks(
    task_type: TaskType,
) -> None:
    """Test subprocess executor keeps provider keys for Flower-controlled tasks."""
    env = {**_PROVIDER_ENV, "PATH": "/usr/bin"}

    with (
        patch.dict(os.environ, env, clear=True),
        patch.object(subprocess, "Popen") as popen_mock,
    ):
        SubprocessExecutor().launch(_execution_spec(task_type=task_type))

    child_env = popen_mock.call_args.kwargs["env"]
    for env_var, value in _PROVIDER_ENV.items():
        assert child_env[env_var] == value


def test_launch_raises_when_subprocess_cannot_start() -> None:
    """Test subprocess executor preserves Popen failure semantics."""
    with patch.object(subprocess, "Popen", side_effect=OSError("missing binary")):
        with pytest.raises(OSError, match="missing binary"):
            SubprocessExecutor().launch(_execution_spec())
