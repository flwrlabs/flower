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
from flwr.supercore.run import Run
from flwr.supercore.superexec.executor import ExecutionSpec
from flwr.supercore.superexec.plugin.base_exec_plugin import BaseExecPlugin
from flwr.supercore.superexec.plugin.clientapp_exec_plugin import ClientAppExecPlugin

from .serverapp_exec_plugin import ServerAppExecPlugin


def _get_run(_: int) -> Run:
    """Return a minimal dummy run."""
    return Run.create_empty(run_id=1)


def _get_task(*, task_id: int = 1, task_type: str = TaskType.CLIENT_APP) -> Mock:
    """Return a minimal dummy task-like object."""
    task = Mock()
    task.task_id = task_id
    task.type = task_type
    return task


def _execution_spec_from_executor(executor: Mock) -> ExecutionSpec:
    """Return the ExecutionSpec passed to a mock executor."""
    return cast(ExecutionSpec, executor.launch.call_args.args[0])


def test_clientapp_launch_delegates_default_stdio_spec() -> None:
    """ClientApp launch should delegate a spec with default stdio behavior."""
    executor = Mock()
    plugin = ClientAppExecPlugin(
        appio_api_address="127.0.0.1:9094",
        insecure=True,
        root_certificates_path=None,
        get_run=_get_run,
        executor=executor,
    )

    plugin.launch_task(token="token", task=_get_task())

    spec = _execution_spec_from_executor(executor)
    assert spec.task_type == TaskType.CLIENT_APP
    assert spec.suppress_output is False


def test_clientapp_launch_ignores_unsupported_task_type() -> None:
    """ClientApp launch should ignore unsupported task types."""
    executor = Mock()
    plugin = ClientAppExecPlugin(
        appio_api_address="127.0.0.1:9094",
        insecure=True,
        root_certificates_path=None,
        get_run=_get_run,
        executor=executor,
    )

    plugin.launch_task(
        token="token", task=_get_task(task_id=5, task_type=TaskType.SERVER_APP)
    )

    executor.launch.assert_not_called()


def test_serverapp_launch_delegates_suppressed_stdio_spec() -> None:
    """ServerApp launch should delegate a spec that suppresses output."""
    executor = Mock()
    plugin = ServerAppExecPlugin(
        appio_api_address="127.0.0.1:9092",
        insecure=True,
        root_certificates_path=None,
        get_run=_get_run,
        executor=executor,
    )

    plugin.launch_task(
        token="token", task=_get_task(task_id=5, task_type=TaskType.SERVER_APP)
    )

    spec = _execution_spec_from_executor(executor)
    assert spec.task_type == TaskType.SERVER_APP
    assert spec.suppress_output is True


def test_simulation_launch_delegates_simulation_task_type() -> None:
    """Simulation launch should delegate a spec with the simulation task type."""
    executor = Mock()
    plugin = ServerAppExecPlugin(
        appio_api_address="127.0.0.1:9092",
        insecure=True,
        root_certificates_path=None,
        get_run=_get_run,
        executor=executor,
    )

    plugin.launch_task(
        token="token", task=_get_task(task_id=5, task_type=TaskType.SIMULATION)
    )

    spec = _execution_spec_from_executor(executor)
    assert spec.task_type == TaskType.SIMULATION
    assert spec.suppress_output is True


class DummyExecPlugin(BaseExecPlugin):
    """Minimal plugin for testing execution spec construction."""

    supported_task_types = frozenset({TaskType.CLIENT_APP})


def test_launch_task_forwards_runtime_dependency_install_flag() -> None:
    """Ensure execution spec forwards runtime install flag."""
    executor = Mock()
    plugin = DummyExecPlugin(
        appio_api_address="127.0.0.1:9091",
        insecure=True,
        root_certificates_path=None,
        get_run=Mock(),
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
        appio_api_address="127.0.0.1:9091",
        insecure=True,
        root_certificates_path=None,
        get_run=Mock(),
        executor=executor,
    )

    plugin.launch_task(token="token-123", task=_get_task(task_id=7))

    assert _execution_spec_from_executor(executor).runtime_dependency_install is False


def test_clientapp_launch_forwards_root_certificate() -> None:
    """ClientApp launch should forward the configured root certificate path."""
    executor = Mock()
    plugin = ClientAppExecPlugin(
        appio_api_address="127.0.0.1:9094",
        insecure=False,
        root_certificates_path="/tmp/root.pem",
        get_run=_get_run,
        executor=executor,
    )

    plugin.launch_task(token="token", task=_get_task(task_id=7))

    spec = _execution_spec_from_executor(executor)
    assert spec.insecure is False
    assert spec.root_certificates_path == "/tmp/root.pem"


def test_clientapp_launch_omits_tls_flags_when_using_system_certificates() -> None:
    """ClientApp launch should omit TLS inputs when relying on system certificates."""
    executor = Mock()
    plugin = ClientAppExecPlugin(
        appio_api_address="127.0.0.1:9094",
        insecure=False,
        root_certificates_path=None,
        get_run=_get_run,
        executor=executor,
    )

    plugin.launch_task(token="token", task=_get_task(task_id=7))

    spec = _execution_spec_from_executor(executor)
    assert spec.insecure is False
    assert spec.root_certificates_path is None
