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
from unittest.mock import patch

from flwr.common.typing import Run
from flwr.supercore.superexec.plugin.clientapp_exec_plugin import ClientAppExecPlugin

from .serverapp_exec_plugin import ServerAppExecPlugin


def _get_run(_: int) -> Run:
    """Return a minimal dummy run."""
    return Run.create_empty(run_id=1)


def test_clientapp_launch_inherits_default_stdio() -> None:
    """ClientApp launch should use default stdio behavior."""
    plugin = ClientAppExecPlugin(
        appio_api_address="127.0.0.1:9094",
        get_run=_get_run,
    )

    with patch("subprocess.Popen") as popen:
        plugin.launch_app(token="token", run_id=7)

    assert "stdout" not in popen.call_args.kwargs
    assert "stderr" not in popen.call_args.kwargs


def test_serverapp_launch_isolates_stdio() -> None:
    """ServerApp launch should not inherit parent stdio streams."""
    plugin = ServerAppExecPlugin(
        appio_api_address="127.0.0.1:9092",
        get_run=_get_run,
    )

    with patch("subprocess.Popen") as popen:
        plugin.launch_app(token="token", run_id=5)

    assert popen.call_args.kwargs["stdout"] is subprocess.DEVNULL
    assert popen.call_args.kwargs["stderr"] is subprocess.DEVNULL
