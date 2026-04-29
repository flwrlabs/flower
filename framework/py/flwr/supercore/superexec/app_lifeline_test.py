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
"""Tests for cooperative SuperExec app lifeline launches."""

# pylint: disable=protected-access

import os
import subprocess
import sys
from unittest.mock import patch

import pytest

from flwr.supercore.superexec.app_lifeline import (
    AppLaunchResult,
    _prepare_popen_kwargs,
    launch_with_lifeline,
)

pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="Lifeline FD launch depends on POSIX FD inheritance.",
)


def test_launch_with_lifeline_wait_passes_open_fd_to_app() -> None:
    """Waiting launch should pass a usable lifeline FD to the app command."""
    returncode = launch_with_lifeline(
        [
            sys.executable,
            "-c",
            (
                "import os, sys; "
                "fd = int(sys.argv[sys.argv.index('--lifeline-fd') + 1]); "
                "os.fstat(fd)"
            ),
        ],
        wait=True,
    )

    assert returncode == 0


def test_launch_with_lifeline_without_wait_returns_app_pid() -> None:
    """Non-waiting launch should return after starting the app process."""
    result = launch_with_lifeline(
        [sys.executable, "-c", "pass"],
        wait=False,
    )

    assert isinstance(result, AppLaunchResult)
    assert result.pid > 0


def test_launch_with_lifeline_preserves_existing_popen_kwargs() -> None:
    """Existing subprocess kwargs should be preserved while adding pass_fds."""
    with patch("flwr.supercore.superexec.app_lifeline.subprocess.Popen") as popen:
        popen.return_value.pid = 123
        popen.return_value.poll.return_value = 0
        launch_with_lifeline(
            [sys.executable, "-c", "pass"],
            wait=False,
            popen_kwargs={
                "pass_fds": (99,),
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
            },
        )

    kwargs = popen.call_args.kwargs
    assert kwargs["close_fds"] is True
    assert 99 in kwargs["pass_fds"]
    assert kwargs["stdout"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.DEVNULL
    assert "--lifeline-fd" in popen.call_args.args[0]


def test_prepare_popen_kwargs_rejects_close_fds_false() -> None:
    """The app must not inherit SuperExec's write end of the lifeline pipe."""
    with pytest.raises(ValueError, match="close_fds=True"):
        _prepare_popen_kwargs({"close_fds": False}, 7)
