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


from unittest.mock import patch

from flwr.common.exit import ExitCode
from flwr.common.typing import Run

from .base_ephemeral_exec_plugin import BaseEphemeralExecPlugin


def _get_run(_: int) -> Run:
    """Return a minimal dummy run."""
    return Run.create_empty(run_id=1)


class _EphemeralExecPlugin(BaseEphemeralExecPlugin):
    command = "flwr-serverapp"
    appio_api_address_arg = "--serverappio-api-address"


def test_select_run_id_returns_none_when_no_candidates() -> None:
    """The plugin should skip execution when no runs are available."""
    plugin = _EphemeralExecPlugin(
        appio_api_address="127.0.0.1:9091",
        get_run=_get_run,
    )

    assert plugin.select_run_id([]) is None


def test_select_run_id_returns_first_candidate() -> None:
    """The plugin should always choose the first candidate run ID."""
    plugin = _EphemeralExecPlugin(
        appio_api_address="127.0.0.1:9091",
        get_run=_get_run,
    )

    assert plugin.select_run_id([7, 9, 11]) == 7


def test_launch_app_runs_expected_command_and_exits() -> None:
    """Launch should invoke the app with token and parent PID, then exit."""
    plugin = _EphemeralExecPlugin(
        appio_api_address="127.0.0.1:9091",
        get_run=_get_run,
    )

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
        plugin.launch_app(token="token-123", run_id=5)

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
