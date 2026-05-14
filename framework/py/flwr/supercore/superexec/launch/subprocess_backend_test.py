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
"""Tests for the SuperExec subprocess launch backend."""


import subprocess
from unittest.mock import patch

from .backend import LaunchSpec
from .subprocess_backend import SubprocessLaunchBackend


def _launch_spec(**overrides: object) -> LaunchSpec:
    """Return a launch spec with sensible test defaults."""
    values = {
        "command": "flwr-clientapp",
        "appio_api_address": "127.0.0.1:9094",
        "appio_api_kind": "clientappio",
        "token": "token",
        "insecure": True,
        "root_certificates_path": None,
        "runtime_dependency_install": False,
        "parent_pid": 1234,
        "suppress_output": False,
    }
    values.update(overrides)
    return LaunchSpec(**values)  # type: ignore[arg-type]


def test_launch_renders_insecure_clientappio_command() -> None:
    """Launch should render insecure ClientAppIo subprocess argv."""
    spec = _launch_spec()

    with patch("subprocess.Popen") as popen:
        SubprocessLaunchBackend().launch(spec)

    assert popen.call_args.args[0] == [
        "flwr-clientapp",
        "--insecure",
        "--clientappio-api-address",
        "127.0.0.1:9094",
        "--token",
        "token",
        "--parent-pid",
        "1234",
    ]
    assert popen.call_args.kwargs == {}


def test_launch_renders_root_certificates() -> None:
    """Launch should render secure subprocess argv with root certificates."""
    spec = _launch_spec(insecure=False, root_certificates_path="/tmp/root.pem")

    with patch("subprocess.Popen") as popen:
        SubprocessLaunchBackend().launch(spec)

    assert popen.call_args.args[0][:3] == [
        "flwr-clientapp",
        "--root-certificates",
        "/tmp/root.pem",
    ]
    assert "--insecure" not in popen.call_args.args[0]


def test_launch_renders_runtime_dependency_install_flag() -> None:
    """Launch should render runtime dependency installation flag."""
    spec = _launch_spec(runtime_dependency_install=True)

    with patch("subprocess.Popen") as popen:
        SubprocessLaunchBackend().launch(spec)

    assert "--allow-runtime-dependency-installation" in popen.call_args.args[0]


def test_launch_omits_parent_pid_when_not_present() -> None:
    """Launch should omit parent PID when the spec does not include one."""
    spec = _launch_spec(parent_pid=None)

    with patch("subprocess.Popen") as popen:
        SubprocessLaunchBackend().launch(spec)

    assert "--parent-pid" not in popen.call_args.args[0]


def test_launch_suppresses_serverapp_output() -> None:
    """Launch should suppress output when requested by the spec."""
    spec = _launch_spec(
        command="flwr-serverapp",
        appio_api_kind="serverappio",
        suppress_output=True,
    )

    with patch("subprocess.Popen") as popen:
        SubprocessLaunchBackend().launch(spec)

    assert "--serverappio-api-address" in popen.call_args.args[0]
    assert popen.call_args.kwargs["stdout"] is subprocess.DEVNULL
    assert popen.call_args.kwargs["stderr"] is subprocess.DEVNULL
