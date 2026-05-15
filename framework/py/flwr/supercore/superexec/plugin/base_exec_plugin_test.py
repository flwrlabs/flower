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

from flwr.common.typing import Run
from flwr.supercore.constant import TaskType
from flwr.supercore.superexec.launch import LaunchSpec
from flwr.supercore.superexec.plugin.base_exec_plugin import BaseExecPlugin
from flwr.supercore.superexec.plugin.clientapp_exec_plugin import ClientAppExecPlugin

from .serverapp_exec_plugin import ServerAppExecPlugin


def _get_run(_: int) -> Run:
    """Return a minimal dummy run."""
    return Run.create_empty(run_id=1)


def _get_task(*, task_id: int = 1, task_type: str = TaskType.SERVER_APP) -> Mock:
    """Return a minimal dummy task-like object."""
    task = Mock()
    task.task_id = task_id
    task.type = task_type
    return task


def _launched_spec(backend: Mock) -> LaunchSpec:
    """Return the LaunchSpec passed to a mock launch backend."""
    return cast(LaunchSpec, backend.launch.call_args.args[0])


def test_clientapp_launch_delegates_default_stdio_spec() -> None:
    """ClientApp launch should delegate a spec with default stdio behavior."""
    backend = Mock()
    plugin = ClientAppExecPlugin(
        appio_api_address="127.0.0.1:9094",
        insecure=True,
        root_certificates_path=None,
        get_run=_get_run,
        launch_backend=backend,
    )

    plugin.launch_task(token="token", task=_get_task())

    spec = _launched_spec(backend)
    assert spec.command == "flwr-clientapp"
    assert spec.appio_api_kind == "clientappio"
    assert spec.suppress_output is False


def test_serverapp_launch_delegates_suppressed_stdio_spec() -> None:
    """ServerApp launch should delegate a spec that suppresses output."""
    backend = Mock()
    plugin = ServerAppExecPlugin(
        appio_api_address="127.0.0.1:9092",
        insecure=True,
        root_certificates_path=None,
        get_run=_get_run,
        launch_backend=backend,
    )

    plugin.launch_task(
        token="token", task=_get_task(task_id=5, task_type=TaskType.SERVER_APP)
    )

    spec = _launched_spec(backend)
    assert spec.command == "flwr-serverapp"
    assert spec.appio_api_kind == "serverappio"
    assert spec.suppress_output is True


class DummyExecPlugin(BaseExecPlugin):
    """Minimal plugin for testing launch spec construction."""

    command = "dummy-app"
    appio_api_kind = "clientappio"


def test_launch_task_forwards_runtime_dependency_install_flag() -> None:
    """Ensure launch spec forwards runtime install flag."""
    backend = Mock()
    plugin = DummyExecPlugin(
        appio_api_address="127.0.0.1:9091",
        insecure=True,
        root_certificates_path=None,
        get_run=Mock(),
        runtime_dependency_install=True,
        launch_backend=backend,
    )

    with patch(
        "flwr.supercore.superexec.plugin.base_exec_plugin.os.getpid",
        return_value=1234,
    ):
        plugin.launch_task(token="token-123", task=_get_task(task_id=7))

    spec = _launched_spec(backend)
    assert spec.runtime_dependency_install is True
    assert spec.parent_pid == 1234


def test_launch_task_skips_optional_runtime_flags_by_default() -> None:
    """Ensure launch spec omits optional runtime install flags by default."""
    backend = Mock()
    plugin = DummyExecPlugin(
        appio_api_address="127.0.0.1:9091",
        insecure=True,
        root_certificates_path=None,
        get_run=Mock(),
        launch_backend=backend,
    )

    plugin.launch_task(token="token-123", task=_get_task(task_id=7))

    assert _launched_spec(backend).runtime_dependency_install is False


def test_clientapp_launch_forwards_root_certificate() -> None:
    """ClientApp launch should forward the configured root certificate path."""
    backend = Mock()
    plugin = ClientAppExecPlugin(
        appio_api_address="127.0.0.1:9094",
        insecure=False,
        root_certificates_path="/tmp/root.pem",
        get_run=_get_run,
        launch_backend=backend,
    )

    plugin.launch_task(token="token", task=_get_task(task_id=7))

    spec = _launched_spec(backend)
    assert spec.insecure is False
    assert spec.root_certificates_path == "/tmp/root.pem"


def test_clientapp_launch_omits_tls_flags_when_using_system_certificates() -> None:
    """ClientApp launch should omit TLS inputs when relying on system certificates."""
    backend = Mock()
    plugin = ClientAppExecPlugin(
        appio_api_address="127.0.0.1:9094",
        insecure=False,
        root_certificates_path=None,
        get_run=_get_run,
        launch_backend=backend,
    )

    plugin.launch_task(token="token", task=_get_task(task_id=7))

    spec = _launched_spec(backend)
    assert spec.insecure is False
    assert spec.root_certificates_path is None
