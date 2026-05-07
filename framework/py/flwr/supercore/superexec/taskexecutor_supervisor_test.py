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
"""Tests for the SuperExec TaskExecutor supervisor."""


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

from flwr.supercore.superexec.taskexecutor_supervisor import (
    _NON_REAPING_WAIT_ATTRS,
    _process_group_exists,
    _run_supervised_process,
    _send_signal_to_process_group,
    _validate_launch_request,
    _validate_termination_grace_period,
    launch_with_lifeline,
)

POSIX_ONLY = pytest.mark.skipif(
    os.name != "posix",
    reason="Lifeline supervision depends on POSIX FD and signal behavior.",
)
LIFELINE_SUPPORTED = pytest.mark.skipif(
    not (
        os.name == "posix"
        and callable(getattr(os, "waitid", None))
        and all(getattr(os, attr, None) is not None for attr in _NON_REAPING_WAIT_ATTRS)
    ),
    reason=(
        "Lifeline supervision depends on POSIX FD/signal behavior and "
        "waitid/WNOWAIT non-reaping child status checks."
    ),
)


@LIFELINE_SUPPORTED
def test_launch_with_lifeline_wait_returns_supervised_process_exit_code() -> None:
    """Waiting launch should return the supervised process exit code."""
    returncode = launch_with_lifeline(
        [sys.executable, "-c", "import sys; sys.exit(7)"],
        wait=True,
        termination_grace_period=0.1,
    )

    assert returncode == 7


@LIFELINE_SUPPORTED
def test_launch_with_lifeline_without_wait_returns_none_after_launch() -> None:
    """Non-waiting launch should return None after command launch is confirmed."""
    result = launch_with_lifeline(
        [sys.executable, "-c", "pass"],
        wait=False,
        termination_grace_period=0.1,
    )

    assert result is None


@LIFELINE_SUPPORTED
def test_launch_with_lifeline_keeps_command_out_of_supervisor_argv() -> None:
    """Command details should travel over the config pipe, not process argv."""

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
        patch(
            "flwr.supercore.superexec.taskexecutor_supervisor.subprocess.Popen",
            _popen,
        ),
        patch("flwr.supercore.superexec.taskexecutor_supervisor._check_launch_status"),
        patch("flwr.supercore.superexec.taskexecutor_supervisor._write_config"),
    ):
        launch_with_lifeline(
            ["flwr-taskexecutor", "--token", "secret-token"],
            wait=False,
        )

    assert "flwr-taskexecutor" not in captured
    assert "secret-token" not in captured


@LIFELINE_SUPPORTED
def test_launch_with_lifeline_reaps_supervisor_after_setup_failure() -> None:
    """Supervisor process should be reaped if parent-side setup fails."""

    class _Popen:
        pid = 1234

        def __init__(self, args: list[str], **_: object) -> None:
            self.args = args
            self.terminated = False
            self.waited = False
            self.returncode: int | None = None

        def poll(self) -> int | None:
            """Return current fake process status."""
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            """Record that the fake process was reaped."""
            del timeout
            self.waited = True
            self.returncode = -signal.SIGTERM
            return self.returncode

        def terminate(self) -> None:
            """Terminate the fake process."""
            self.terminated = True

        def kill(self) -> None:
            """Kill the fake process."""
            raise AssertionError("kill should not be needed when wait succeeds")

    fake_supervisor: _Popen | None = None

    def _popen(args: list[str], **kwargs: object) -> _Popen:
        del kwargs
        nonlocal fake_supervisor
        fake_supervisor = _Popen(args)
        return fake_supervisor

    with (
        patch(
            "flwr.supercore.superexec.taskexecutor_supervisor.subprocess.Popen",
            _popen,
        ),
        patch(
            "flwr.supercore.superexec.taskexecutor_supervisor._write_config",
            side_effect=RuntimeError("config write failed"),
        ),
        pytest.raises(RuntimeError, match="config write failed"),
    ):
        launch_with_lifeline([sys.executable, "-c", "pass"], wait=False)

    assert fake_supervisor is not None
    assert fake_supervisor.terminated
    assert fake_supervisor.waited


@LIFELINE_SUPPORTED
def test_launch_with_lifeline_reports_command_launch_failure(tmp_path: Path) -> None:
    """Non-waiting launch should fail if the supervisor cannot launch the command."""
    missing_command = tmp_path / "missing-command"

    with pytest.raises(RuntimeError, match="failed to launch command"):
        launch_with_lifeline(
            [str(missing_command)],
            wait=False,
            termination_grace_period=0.1,
        )


@LIFELINE_SUPPORTED
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


@LIFELINE_SUPPORTED
def test_lifeline_closure_terminates_supervised_process_group(
    tmp_path: Path,
) -> None:
    """Closing the lifeline should terminate the supervised process."""
    pid_file = tmp_path / "supervised.pid"
    read_fd, write_fd = os.pipe()
    closer = threading.Thread(
        target=_close_fd_when_file_exists,
        args=(write_fd, pid_file),
        daemon=True,
    )
    closer.start()

    try:
        returncode = _run_supervised_process(
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


@LIFELINE_SUPPORTED
def test_lifeline_closure_escalates_to_sigkill(tmp_path: Path) -> None:
    """Supervisor should escalate if the supervised process ignores SIGTERM."""
    pid_file = tmp_path / "supervised.pid"
    read_fd, write_fd = os.pipe()
    closer = threading.Thread(
        target=_close_fd_when_file_exists,
        args=(write_fd, pid_file),
        daemon=True,
    )
    closer.start()

    try:
        returncode = _run_supervised_process(
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


@LIFELINE_SUPPORTED
def test_process_exit_cleans_remaining_process_group_children(tmp_path: Path) -> None:
    """Supervisor should clean children left behind by the supervised command."""
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
        returncode = _run_supervised_process(
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


@LIFELINE_SUPPORTED
def test_supervised_process_does_not_inherit_lifeline_fd() -> None:
    """The supervised process should not inherit the supervisor lifeline FD."""
    read_fd, write_fd = os.pipe()
    env = os.environ.copy()
    env["LIFELINE_FD"] = str(read_fd)

    try:
        returncode = _run_supervised_process(
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


def test_validate_launch_request_rejects_circular_popen_kwargs() -> None:
    """Circular kwargs should get the normalized JSON-serializable error."""
    popen_kwargs: dict[str, Any] = {}
    popen_kwargs["self"] = popen_kwargs

    with pytest.raises(TypeError, match="must be JSON serializable"):
        _validate_launch_request([sys.executable, "-c", "pass"], popen_kwargs)


def test_validate_launch_request_rejects_unavailable_stdio_fd() -> None:
    """Arbitrary stdio FDs should not be passed through the supervisor config."""
    with pytest.raises(ValueError, match="unsupported stdio"):
        _validate_launch_request([sys.executable, "-c", "pass"], {"stdout": 99})


def test_validate_termination_grace_period_rejects_negative_values() -> None:
    """Negative grace periods should not silently mean immediate SIGKILL."""
    with pytest.raises(ValueError, match="finite non-negative"):
        _validate_termination_grace_period(-1.0)


@POSIX_ONLY
def test_launch_with_lifeline_rejects_missing_non_reaping_wait_support() -> None:
    """Supervisor should fail before launch if waitid support is unavailable."""
    with (
        patch(
            "flwr.supercore.superexec.taskexecutor_supervisor.os.waitid",
            None,
            create=True,
        ),
        patch(
            "flwr.supercore.superexec.taskexecutor_supervisor.subprocess.Popen",
        ) as popen,
        pytest.raises(RuntimeError, match="waitid"),
    ):
        launch_with_lifeline([sys.executable, "-c", "pass"], wait=False)

    popen.assert_not_called()


@POSIX_ONLY
def test_process_group_exists_raises_on_permission_error() -> None:
    """Permission errors should not be treated as missing process groups."""
    with (
        patch(
            "flwr.supercore.superexec.taskexecutor_supervisor.os.killpg",
            side_effect=PermissionError,
        ),
        pytest.raises(RuntimeError, match="permission denied"),
    ):
        _process_group_exists(1234)


@POSIX_ONLY
def test_send_signal_to_process_group_raises_on_permission_error() -> None:
    """Permission errors while signaling should be surfaced to callers."""
    with (
        patch(
            "flwr.supercore.superexec.taskexecutor_supervisor.os.killpg",
            side_effect=PermissionError,
        ),
        pytest.raises(RuntimeError, match="permission denied"),
    ):
        _send_signal_to_process_group(1234, signal.SIGTERM)


def test_launch_with_lifeline_rejects_non_posix_platform() -> None:
    """The public launch helper should fail clearly outside POSIX platforms."""
    with (
        patch("flwr.supercore.superexec.taskexecutor_supervisor.os.name", "nt"),
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
