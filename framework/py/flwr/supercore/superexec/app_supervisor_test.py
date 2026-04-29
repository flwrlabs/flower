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
"""Tests for the SuperExec app supervisor."""


# pylint: disable=protected-access

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

from flwr.supercore.superexec.app_supervisor import (
    AppLaunchResult,
    _run_supervised_app,
    _validate_launch_request,
    _validate_termination_grace_period,
    launch_with_lifeline,
)

POSIX_ONLY = pytest.mark.skipif(
    os.name != "posix",
    reason="Lifeline supervision depends on POSIX FD and signal behavior.",
)


@POSIX_ONLY
def test_launch_with_lifeline_wait_returns_app_exit_code() -> None:
    """Waiting launch should return the supervised app exit code."""
    returncode = launch_with_lifeline(
        [sys.executable, "-c", "import sys; sys.exit(7)"],
        wait=True,
        termination_grace_period=0.1,
    )

    assert returncode == 7


@POSIX_ONLY
def test_launch_with_lifeline_without_wait_returns_supervisor_pid() -> None:
    """Non-waiting launch should return after starting the supervisor."""
    result = launch_with_lifeline(
        [sys.executable, "-c", "pass"],
        wait=False,
        termination_grace_period=0.1,
    )

    assert isinstance(result, AppLaunchResult)
    assert result.supervisor_pid > 0


@POSIX_ONLY
def test_launch_with_lifeline_keeps_app_command_out_of_supervisor_argv() -> None:
    """App command details should travel over the config pipe, not process argv."""

    class _Popen:
        pid = 1234

        def __init__(self, args: list[str], **_: object) -> None:
            self.args = args
            self.terminated = False

        def poll(self) -> int | None:
            """Return a completed process status."""
            return 0

        def wait(self) -> int:
            """Return a completed process status."""
            return 0

        def terminate(self) -> None:
            """Terminate the fake process."""
            self.terminated = True

    captured: list[str] = []

    def _popen(args: list[str], **kwargs: object) -> _Popen:
        del kwargs
        captured.extend(args)
        return _Popen(args)

    with (
        patch("flwr.supercore.superexec.app_supervisor.subprocess.Popen", _popen),
        patch("flwr.supercore.superexec.app_supervisor._check_launch_status"),
        patch("flwr.supercore.superexec.app_supervisor._write_config"),
    ):
        launch_with_lifeline(
            ["dummy-app", "--token", "secret-token"],
            wait=False,
        )

    assert "dummy-app" not in captured
    assert "secret-token" not in captured


@POSIX_ONLY
def test_launch_with_lifeline_reports_app_launch_failure(tmp_path: Path) -> None:
    """Non-waiting launch should fail if the supervisor cannot launch the app."""
    missing_command = tmp_path / "missing-command"

    with pytest.raises(RuntimeError, match="failed to launch app command"):
        launch_with_lifeline(
            [str(missing_command)],
            wait=False,
            termination_grace_period=0.1,
        )


@POSIX_ONLY
def test_launch_with_lifeline_preserves_devnull_stdio() -> None:
    """DEVNULL stdio kwargs should survive the supervisor config pipe."""
    returncode = launch_with_lifeline(
        [
            sys.executable,
            "-c",
            "import sys; print('out'); print('err', file=sys.stderr)",
        ],
        wait=True,
        popen_kwargs={"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL},
        termination_grace_period=0.1,
    )

    assert returncode == 0


@POSIX_ONLY
def test_lifeline_closure_terminates_app_process_group(tmp_path: Path) -> None:
    """Closing the lifeline should terminate the supervised app."""
    pid_file = tmp_path / "app.pid"
    read_fd, write_fd = os.pipe()
    closer = threading.Thread(
        target=_close_fd_when_file_exists,
        args=(write_fd, pid_file),
        daemon=True,
    )
    closer.start()

    try:
        returncode = _run_supervised_app(
            [
                sys.executable,
                "-c",
                (
                    "import os, pathlib, sys, time; "
                    "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
                    "time.sleep(30)"
                ),
                str(pid_file),
            ],
            lifeline_fd=read_fd,
            popen_kwargs={},
            termination_grace_period=0.1,
        )
    finally:
        _close_fd(read_fd)
        _close_fd(write_fd)

    assert returncode < 0


@POSIX_ONLY
def test_lifeline_closure_escalates_to_sigkill(tmp_path: Path) -> None:
    """Supervisor should escalate if the app ignores SIGTERM."""
    pid_file = tmp_path / "app.pid"
    read_fd, write_fd = os.pipe()
    closer = threading.Thread(
        target=_close_fd_when_file_exists,
        args=(write_fd, pid_file),
        daemon=True,
    )
    closer.start()

    try:
        returncode = _run_supervised_app(
            [
                sys.executable,
                "-c",
                (
                    "import os, pathlib, signal, sys, time; "
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                    "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
                    "time.sleep(30)"
                ),
                str(pid_file),
            ],
            lifeline_fd=read_fd,
            popen_kwargs={},
            termination_grace_period=0.1,
        )
    finally:
        _close_fd(read_fd)
        _close_fd(write_fd)

    assert returncode == -signal.SIGKILL


@POSIX_ONLY
def test_app_exit_cleans_remaining_process_group_children(tmp_path: Path) -> None:
    """Supervisor should clean app children left behind by the app command."""
    ready_file = tmp_path / "child.ready"
    terminated_file = tmp_path / "child.terminated"
    read_fd, write_fd = os.pipe()
    child_code = (
        "import pathlib, signal, sys, time\n"
        f"terminated_file = pathlib.Path({str(terminated_file)!r})\n"
        "def handle_sigterm(_signum, _frame):\n"
        "    terminated_file.write_text('done')\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, handle_sigterm)\n"
        f"pathlib.Path({str(ready_file)!r}).write_text('ready')\n"
        "time.sleep(30)\n"
    )
    parent_code = (
        "import pathlib, subprocess, sys, time\n"
        f"ready_file = pathlib.Path({str(ready_file)!r})\n"
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        "deadline = time.monotonic() + 5\n"
        "while time.monotonic() < deadline and not ready_file.exists():\n"
        "    time.sleep(0.01)\n"
    )

    try:
        returncode = _run_supervised_app(
            [sys.executable, "-c", parent_code],
            lifeline_fd=read_fd,
            popen_kwargs={},
            termination_grace_period=1.0,
        )
    finally:
        _close_fd(read_fd)
        _close_fd(write_fd)

    assert returncode == 0
    assert terminated_file.exists()


@POSIX_ONLY
def test_app_process_does_not_inherit_lifeline_fd() -> None:
    """The supervised app should not inherit the supervisor lifeline FD."""
    read_fd, write_fd = os.pipe()
    env = os.environ.copy()
    env["LIFELINE_FD"] = str(read_fd)

    try:
        returncode = _run_supervised_app(
            [
                sys.executable,
                "-c",
                (
                    "import os, sys; "
                    "fd = int(os.environ['LIFELINE_FD']); "
                    "\ntry:\n    os.fstat(fd)\nexcept OSError:\n    sys.exit(0)\n"
                    "sys.exit(44)"
                ),
            ],
            lifeline_fd=read_fd,
            popen_kwargs={"env": env},
            termination_grace_period=0.1,
        )
    finally:
        _close_fd(read_fd)
        _close_fd(write_fd)

    assert returncode == 0


def test_validate_launch_request_rejects_lifecycle_kwargs() -> None:
    """Lifecycle-owned Popen kwargs should be rejected."""
    with pytest.raises(ValueError, match="lifecycle-owned"):
        _validate_launch_request(
            [sys.executable, "-c", "pass"],
            {"start_new_session": False},
        )


def test_validate_launch_request_rejects_non_string_popen_kwargs_keys() -> None:
    """Popen kwargs keys should not rely on JSON key coercion."""
    with pytest.raises(TypeError, match="keys must be strings"):
        _validate_launch_request(
            [sys.executable, "-c", "pass"],
            cast(Any, {1: "value"}),
        )


def test_validate_launch_request_accepts_devnull_stdio() -> None:
    """Existing ServerApp stdio isolation should remain supported."""
    _validate_launch_request(
        [sys.executable, "-c", "pass"],
        {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL},
    )


def test_validate_launch_request_rejects_unavailable_stdio_fd() -> None:
    """Arbitrary stdio FDs should not be passed through the supervisor config."""
    with pytest.raises(ValueError, match="unsupported stdio"):
        _validate_launch_request([sys.executable, "-c", "pass"], {"stdout": 99})


def test_validate_termination_grace_period_rejects_negative_values() -> None:
    """Negative grace periods should not silently mean immediate SIGKILL."""
    with pytest.raises(ValueError, match="finite non-negative"):
        _validate_termination_grace_period(-1.0)


def test_launch_with_lifeline_rejects_non_posix_platform() -> None:
    """The public launch helper should fail clearly outside POSIX platforms."""
    with (
        patch("flwr.supercore.superexec.app_supervisor.os.name", "nt"),
        pytest.raises(RuntimeError, match="requires POSIX"),
    ):
        launch_with_lifeline([sys.executable, "-c", "pass"], wait=True)


def _close_fd_when_file_exists(fd: int, path: Path) -> None:
    """Close an FD once a child writes its readiness file."""
    deadline = time.monotonic() + 5
    try:
        while time.monotonic() < deadline:
            if path.exists():
                return
            time.sleep(0.01)
    finally:
        _close_fd(fd)


def _close_fd(fd: int) -> None:
    """Close an FD, ignoring already-closed descriptors."""
    try:
        os.close(fd)
    except OSError:
        pass
