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


import subprocess
from unittest.mock import Mock, patch

from flwr.common.typing import Run
from flwr.supercore.constant import TaskType
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


def test_clientapp_launch_inherits_default_stdio() -> None:
    """ClientApp launch should use default stdio behavior."""
    plugin = ClientAppExecPlugin(
        appio_api_address="127.0.0.1:9094",
        insecure=True,
        root_certificates_path=None,
        get_run=_get_run,
    )

    with patch("subprocess.Popen") as popen:
        plugin.launch_task(token="token", task=_get_task())

    assert "stdout" not in popen.call_args.kwargs
    assert "stderr" not in popen.call_args.kwargs


def test_serverapp_launch_isolates_stdio() -> None:
    """ServerApp launch should not inherit parent stdio streams."""
    plugin = ServerAppExecPlugin(
        appio_api_address="127.0.0.1:9092",
        insecure=True,
        root_certificates_path=None,
        get_run=_get_run,
    )

    with patch("subprocess.Popen") as popen:
        plugin.launch_task(
            token="token", task=_get_task(task_id=5, task_type=TaskType.SERVER_APP)
        )

    assert popen.call_args.kwargs["stdout"] is subprocess.DEVNULL
    assert popen.call_args.kwargs["stderr"] is subprocess.DEVNULL


class DummyExecPlugin(BaseExecPlugin):
    """Minimal plugin for testing command construction."""

    command = "dummy-app"
    appio_api_address_arg = "--appio-api-address"


def test_launch_task_forwards_runtime_dependency_install_flag() -> None:
    """Ensure app launch forwards runtime install flag."""
    plugin = DummyExecPlugin(
        appio_api_address="127.0.0.1:9091",
        insecure=True,
        root_certificates_path=None,
        get_run=Mock(),
        runtime_dependency_install=True,
    )

    with (
        patch(
            "flwr.supercore.superexec.plugin.base_exec_plugin.os.getpid",
            return_value=1234,
        ),
        patch(
            "flwr.supercore.superexec.plugin.base_exec_plugin.subprocess.Popen"
        ) as popen,
    ):
        plugin.launch_task(token="token-123", task=_get_task(task_id=7))

    assert popen.call_args.args[0] == [
        "dummy-app",
        "--insecure",
        "--appio-api-address",
        "127.0.0.1:9091",
        "--token",
        "token-123",
        "--parent-pid",
        "1234",
        "--allow-runtime-dependency-installation",
    ]


def test_launch_task_skips_optional_runtime_flags_by_default() -> None:
    """Ensure app launch omits optional runtime install flags by default."""
    plugin = DummyExecPlugin(
        appio_api_address="127.0.0.1:9091",
        insecure=True,
        root_certificates_path=None,
        get_run=Mock(),
    )

    with patch(
        "flwr.supercore.superexec.plugin.base_exec_plugin.subprocess.Popen"
    ) as popen:
        plugin.launch_task(token="token-123", task=_get_task(task_id=7))

    assert "--allow-runtime-dependency-installation" not in popen.call_args.args[0]


def test_clientapp_launch_forwards_root_certificate() -> None:
    """ClientApp launch should forward the configured root certificate path."""
    plugin = ClientAppExecPlugin(
        appio_api_address="127.0.0.1:9094",
        insecure=False,
        root_certificates_path="/tmp/root.pem",
        get_run=_get_run,
    )

    with patch("subprocess.Popen") as mock_popen:
        plugin.launch_task(token="token", task=_get_task(task_id=7))

    assert mock_popen.call_args.args[0][:3] == [
        "flwr-clientapp",
        "--root-certificates",
        "/tmp/root.pem",
    ]


def test_clientapp_launch_omits_tls_flags_when_using_system_certificates() -> None:
    """ClientApp launch should omit TLS flags when relying on system certificates."""
    plugin = ClientAppExecPlugin(
        appio_api_address="127.0.0.1:9094",
        insecure=False,
        root_certificates_path=None,
        get_run=_get_run,
    )

    with patch("subprocess.Popen") as mock_popen:
        plugin.launch_task(token="token", task=_get_task(task_id=7))

    assert "--insecure" not in mock_popen.call_args.args[0]
    assert "--root-certificates" not in mock_popen.call_args.args[0]
