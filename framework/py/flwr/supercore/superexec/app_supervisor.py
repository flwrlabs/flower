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
"""Trusted local supervisor for SuperExec-launched app processes."""

import argparse
import json
import os
import selectors
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_LIFECYCLE_POPEN_KWARGS = frozenset(
    {
        "close_fds",
        "pass_fds",
        "preexec_fn",
        "process_group",
        "start_new_session",
    }
)
_READ_SIZE = 65536
_POLL_INTERVAL = 0.1
_STDIO_POPEN_KWARGS = frozenset({"stdin", "stdout", "stderr"})


@dataclass(frozen=True)
class AppLaunchResult:
    """Result for a non-waiting app launch."""

    supervisor_pid: int


def launch_with_lifeline(
    command: list[str],
    *,
    wait: bool,
    popen_kwargs: dict[str, Any] | None = None,
    termination_grace_period: float = 5.0,
) -> AppLaunchResult | int:
    """Launch an app command through a supervisor with a lifeline FD.

    When ``wait`` is ``False``, this returns after the supervisor starts. When
    ``wait`` is ``True``, this waits for the supervisor and returns its exit code.
    ``popen_kwargs`` must be JSON-serializable because the launch request is sent to
    the supervisor over a config pipe.
    """
    if os.name != "posix":
        raise RuntimeError("lifeline supervision requires POSIX FD inheritance")
    _validate_launch_request(command, popen_kwargs)
    lifeline_read_fd, lifeline_write_fd = os.pipe()
    config_read_fd, config_write_fd = os.pipe()
    supervisor: subprocess.Popen[bytes] | None = None

    try:
        # Keep app command details out of supervisor argv so token-bearing launch
        # commands are not exposed through process listings.
        supervisor_command = [
            sys.executable,
            "-m",
            "flwr.supercore.superexec.app_supervisor",
            "--lifeline-fd",
            str(lifeline_read_fd),
            "--config-fd",
            str(config_read_fd),
            "--termination-grace-period",
            str(termination_grace_period),
        ]
        supervisor = subprocess.Popen(  # pylint: disable=consider-using-with
            supervisor_command,
            close_fds=True,
            env=_supervisor_env(),
            pass_fds=(lifeline_read_fd, config_read_fd),
        )
        os.close(lifeline_read_fd)
        lifeline_read_fd = -1
        os.close(config_read_fd)
        config_read_fd = -1

        _write_config(config_write_fd, command, popen_kwargs or {})
        os.close(config_write_fd)
        config_write_fd = -1

        if wait:
            # Keep the lifeline open while SuperExec waits so normal app exit is not
            # misinterpreted as parent death by the supervisor.
            try:
                return supervisor.wait()
            finally:
                _close_fd(lifeline_write_fd)
                lifeline_write_fd = -1

        _start_supervisor_reaper(supervisor, lifeline_write_fd)
        lifeline_write_fd = -1
        return AppLaunchResult(supervisor_pid=supervisor.pid)
    except Exception:
        if supervisor is not None and supervisor.poll() is None:
            supervisor.terminate()
        raise
    finally:
        _close_fd(lifeline_read_fd)
        _close_fd(lifeline_write_fd)
        _close_fd(config_read_fd)
        _close_fd(config_write_fd)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the app supervisor entrypoint."""
    parser = argparse.ArgumentParser(description="Run a supervised Flower app command.")
    parser.add_argument("--lifeline-fd", type=int, required=True)
    parser.add_argument("--config-fd", type=int, required=True)
    parser.add_argument("--termination-grace-period", type=float, default=5.0)
    args = parser.parse_args(argv)
    config_fd = args.config_fd
    lifeline_fd = args.lifeline_fd

    try:
        os.set_inheritable(lifeline_fd, False)
        config = _read_config(config_fd)
        _close_fd(config_fd)
        config_fd = -1
        command = config["command"]
        popen_kwargs = config["popen_kwargs"]
        _validate_launch_request(command, popen_kwargs)
        return _run_supervised_app(
            command,
            lifeline_fd=lifeline_fd,
            popen_kwargs=popen_kwargs,
            termination_grace_period=args.termination_grace_period,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"Failed to supervise app command: {exc}", file=sys.stderr)
        return 2
    finally:
        _close_fd(config_fd)
        _close_fd(lifeline_fd)


def _run_supervised_app(
    command: list[str],
    *,
    lifeline_fd: int,
    popen_kwargs: dict[str, Any],
    termination_grace_period: float,
) -> int:
    """Launch and supervise the app command."""
    app_process = subprocess.Popen(  # pylint: disable=consider-using-with
        command,
        **popen_kwargs,
        start_new_session=True,
        close_fds=True,
    )
    process_group_id = app_process.pid
    try:
        while True:
            returncode = app_process.poll()
            if returncode is not None:
                return returncode
            if _lifeline_closed(lifeline_fd):
                # EOF on the pipe means SuperExec exited or deliberately closed its
                # control end; cleanup is enforced outside any app PID namespace.
                _terminate_process_group(
                    process_group_id,
                    app_process,
                    termination_grace_period,
                )
                return app_process.wait()
            time.sleep(_POLL_INTERVAL)
    finally:
        # The wrapper/app can leave same-process-group children after the leader exits;
        # cleanup the whole group even on normal return.
        _terminate_process_group(
            process_group_id,
            app_process,
            termination_grace_period,
        )


def _lifeline_closed(lifeline_fd: int) -> bool:
    """Return True if the lifeline FD has reached EOF."""
    selector = selectors.DefaultSelector()
    try:
        selector.register(lifeline_fd, selectors.EVENT_READ)
        events = selector.select(timeout=0)
    finally:
        selector.close()
    if not events:
        return False
    return os.read(lifeline_fd, 1) == b""


def _terminate_process_group(
    process_group_id: int,
    app_process: subprocess.Popen[bytes],
    grace_period: float,
) -> None:
    """Terminate the app process group, escalating to SIGKILL if necessary."""
    if not _process_group_exists(process_group_id):
        return

    _send_signal_to_process_group(process_group_id, signal.SIGTERM)
    deadline = time.monotonic() + grace_period
    while time.monotonic() < deadline:
        app_process.poll()
        if not _process_group_exists(process_group_id):
            return
        time.sleep(_POLL_INTERVAL)
    if _process_group_exists(process_group_id):
        _send_signal_to_process_group(process_group_id, signal.SIGKILL)


def _send_signal_to_process_group(process_group_id: int, sig: signal.Signals) -> None:
    """Send a signal to a process group, ignoring already-exited groups."""
    try:
        os.killpg(process_group_id, sig)
    except (PermissionError, ProcessLookupError):
        return


def _process_group_exists(process_group_id: int) -> bool:
    """Return True if a process group still exists."""
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return False
    return True


def _supervisor_env() -> dict[str, str]:
    """Return an environment where this package is importable by ``python -m``."""
    env = os.environ.copy()
    package_parent = str(Path(__file__).resolve().parents[3])
    python_path = env.get("PYTHONPATH")
    if python_path:
        env["PYTHONPATH"] = f"{package_parent}{os.pathsep}{python_path}"
    else:
        env["PYTHONPATH"] = package_parent
    return env


def _validate_launch_request(
    command: list[str],
    popen_kwargs: dict[str, Any] | None,
) -> None:
    """Validate command and kwargs before handing them to subprocess."""
    if not command or not all(isinstance(part, str) for part in command):
        raise ValueError("command must be a non-empty list of strings")
    if popen_kwargs is None:
        return
    if not isinstance(popen_kwargs, dict):
        raise TypeError("popen_kwargs must be a dictionary")
    lifecycle_kwargs = _LIFECYCLE_POPEN_KWARGS.intersection(popen_kwargs)
    if lifecycle_kwargs:
        rejected = ", ".join(sorted(lifecycle_kwargs))
        raise ValueError(f"popen_kwargs includes lifecycle-owned keys: {rejected}")
    # The config pipe intentionally supports only simple, serializable subprocess
    # settings. Current callers only need DEVNULL stdio preservation; arbitrary FDs
    # would need explicit FD passing semantics and are rejected instead.
    unsupported_stdio = {
        key: value
        for key, value in popen_kwargs.items()
        if key in _STDIO_POPEN_KWARGS
        and value is not None
        and value != subprocess.DEVNULL
    }
    if unsupported_stdio:
        rejected = ", ".join(sorted(unsupported_stdio))
        raise ValueError(f"popen_kwargs includes unsupported stdio keys: {rejected}")
    try:
        json.dumps({"command": command, "popen_kwargs": popen_kwargs})
    except TypeError as exc:
        raise TypeError("popen_kwargs must be JSON serializable") from exc


def _write_config(
    config_write_fd: int,
    command: list[str],
    popen_kwargs: dict[str, Any],
) -> None:
    """Write launch config to the supervisor config pipe."""
    config_bytes = json.dumps(
        {"command": command, "popen_kwargs": popen_kwargs},
        separators=(",", ":"),
    ).encode("utf-8")
    while config_bytes:
        bytes_written = os.write(config_write_fd, config_bytes)
        config_bytes = config_bytes[bytes_written:]


def _read_config(config_read_fd: int) -> dict[str, Any]:
    """Read launch config from the parent config pipe."""
    chunks = []
    while True:
        chunk = os.read(config_read_fd, _READ_SIZE)
        if not chunk:
            break
        chunks.append(chunk)
    config = json.loads(b"".join(chunks).decode("utf-8"))
    if not isinstance(config, dict):
        raise TypeError("supervisor config must be a dictionary")
    return config


def _start_supervisor_reaper(
    supervisor: subprocess.Popen[bytes],
    lifeline_write_fd: int,
) -> None:
    """Close the lifeline control FD after the supervisor exits."""

    def reap() -> None:
        try:
            supervisor.wait()
        finally:
            _close_fd(lifeline_write_fd)

    threading.Thread(target=reap, daemon=True).start()


def _close_fd(fd: int) -> None:
    """Close an FD if it is open."""
    if fd < 0:
        return
    try:
        os.close(fd)
    except OSError:
        return


if __name__ == "__main__":
    sys.exit(main())
