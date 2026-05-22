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


import subprocess
from typing import Any
from unittest.mock import Mock, patch

from flwr.supercore.constant import TaskType

from .subprocess_executor import SubprocessExecutor
from .types import ExecutionSpec


def _execution_spec(**overrides: Any) -> ExecutionSpec:
    base: dict[str, Any] = {
        "task_type": TaskType.CLIENT_APP,
        "appio_api_address": "127.0.0.1:9094",
        "token": "token",
        "insecure": True,
        "root_certificates_path": None,
        "runtime_dependency_install": False,
        "parent_pid": None,
        "suppress_output": False,
    }
    base.update(overrides)
    return ExecutionSpec(**base)


def test_launch_renders_insecure_clientapp_args() -> None:
    """Test subprocess executor renders insecure ClientApp args."""
    with patch.object(subprocess, "Popen") as popen_mock:
        SubprocessExecutor().launch(_execution_spec())

    popen_mock.assert_called_once_with(
        [
            "flwr-clientapp",
            "--clientappio-api-address",
            "127.0.0.1:9094",
            "--token",
            "token",
            "--insecure",
        ]
    )


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
            "--clientappio-api-address",
            "127.0.0.1:9094",
            "--token",
            "token",
            "--root-certificates",
            "/path/to/root.pem",
        ]
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
        SubprocessExecutor().launch(
            _execution_spec(
                task_type=TaskType.SERVER_APP,
                suppress_output=True,
            )
        )

    popen_mock.assert_called_once_with(
        [
            "flwr-serverapp",
            "--serverappio-api-address",
            "127.0.0.1:9094",
            "--token",
            "token",
            "--insecure",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_launch_renders_simulation_args() -> None:
    """Test subprocess executor renders Simulation args."""
    with patch.object(subprocess, "Popen") as popen_mock:
        SubprocessExecutor().launch(_execution_spec(task_type=TaskType.SIMULATION))

    popen_mock.assert_called_once_with(
        [
            "flwr-simulation",
            "--serverappio-api-address",
            "127.0.0.1:9094",
            "--token",
            "token",
            "--insecure",
        ]
    )


def test_launch_does_not_suppress_output_by_default() -> None:
    """Test subprocess executor leaves output inherited by default."""
    popen_mock = Mock()

    with patch.object(subprocess, "Popen", popen_mock):
        SubprocessExecutor().launch(_execution_spec())

    assert "stdout" not in popen_mock.call_args.kwargs
    assert "stderr" not in popen_mock.call_args.kwargs
