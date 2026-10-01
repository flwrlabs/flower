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
"""Tests for SuperExec base plugin launch behavior."""

from typing import cast
from unittest.mock import Mock, patch

from flwr.supercore.constant import TaskType
from flwr.supercore.superexec.executor import ExecutionSpec
from flwr.supercore.superexec.plugin.base_exec_plugin import (
    AutoExecPlugin,
    BaseExecPlugin,
)


def _get_task(
    *,
    task_id: int = 1,
    task_type: str = TaskType.CLIENT_APP,
    fab_hash: str | None = None,
) -> Mock:
    """Return a minimal dummy task-like object."""
    task = Mock()
    task.task_id = task_id
    task.type = task_type
    task.fab_hash = fab_hash
    return task


def _execution_spec_from_executor(executor: Mock) -> ExecutionSpec:
    """Return the ExecutionSpec passed to a mock executor."""
    return cast(ExecutionSpec, executor.launch.call_args.args[0])


def test_default_plugin_uses_task_type() -> None:
    """Default plugin should pass task type and FAB identity to the executor."""
    executor = Mock()
    plugin = AutoExecPlugin(
        runtime_api_address="127.0.0.1:9091",
        insecure=True,
        root_certificates_path=None,
        executor=executor,
    )

    plugin.launch_task(
        token="token",
        task=_get_task(task_type=TaskType.AGENT_APP, fab_hash="fab-hash"),
    )

    spec = _execution_spec_from_executor(executor)
    assert spec.task_type == TaskType.AGENT_APP
    assert spec.fab_hash == "fab-hash"


class DummyExecPlugin(BaseExecPlugin):
    """Minimal plugin for testing execution spec construction."""

    supported_task_types = frozenset({TaskType.CLIENT_APP})


def test_launch_task_forwards_runtime_dependency_install_flag() -> None:
    """Ensure execution spec forwards runtime install flag."""
    executor = Mock()
    plugin = DummyExecPlugin(
        runtime_api_address="127.0.0.1:9091",
        insecure=True,
        root_certificates_path=None,
        runtime_dependency_install=True,
        executor=executor,
    )

    with patch(
        "flwr.supercore.superexec.plugin.base_exec_plugin.os.getpid",
        return_value=1234,
    ):
        plugin.launch_task(token="token-123", task=_get_task(task_id=7))

    spec = _execution_spec_from_executor(executor)
    assert spec.runtime_dependency_install is True
    assert spec.parent_pid == 1234
    assert spec.task_id == 7


def test_launch_task_skips_optional_runtime_flags_by_default() -> None:
    """Ensure execution spec omits optional runtime install flags by default."""
    executor = Mock()
    plugin = DummyExecPlugin(
        runtime_api_address="127.0.0.1:9091",
        insecure=True,
        root_certificates_path=None,
        executor=executor,
    )

    plugin.launch_task(token="token-123", task=_get_task(task_id=7))

    assert _execution_spec_from_executor(executor).runtime_dependency_install is False
