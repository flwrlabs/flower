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
"""Trusted local supervisor for SuperExec-launched TaskExecutor processes.

SuperExec starts this module as a small host-side lifecycle process, then sends the real
TaskExecutor command over a config pipe so token-bearing argv entries are not visible in
the supervisor process listing. The supervisor starts that command in a new process
group, reports launch success over a status pipe, and watches a parent-owned lifeline
FD. When the lifeline reaches EOF, the supervisor terminates the TaskExecutor process
group from outside any TaskExecutor-visible PID namespace.
"""


import argparse
import json
import math
import os
import selectors
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

# These Popen kwargs are owned by the supervisor to preserve FD isolation and
# process-group control, so callers may not override exactly these lifecycle hooks.
_LIFECYCLE_POPEN_KWARGS = frozenset(
    {
        "close_fds",
        "pass_fds",
        "preexec_fn",
        "process_group",
        "start_new_session",
    }
)
# Read config/status pipes in bounded chunks; 64 KiB keeps reads efficient without
# assuming the JSON payload fits in a single OS pipe buffer.
_READ_SIZE = 65536
# Poll often enough for prompt cleanup while avoiding a busy loop in the supervisor.
_POLL_INTERVAL = 0.1
# Failed setup should not leave a supervisor process behind; one second gives normal
# termination a chance before escalating to kill.
_SUPERVISOR_STOP_TIMEOUT = 1.0
# Only stdio kwargs need special validation today, and these are the subprocess stdio
# keys whose FD semantics would otherwise cross the config-pipe boundary.
_STDIO_POPEN_KWARGS = frozenset({"stdin", "stdout", "stderr"})
# All of these os.waitid names are required to check child status without reaping;
# missing any one means the platform cannot safely avoid the PID/PGID reuse hazard.
_NON_REAPING_WAIT_ATTRS = (
    "P_PID",
    "WEXITED",
    "WNOHANG",
    "WNOWAIT",
    "CLD_EXITED",
    "CLD_KILLED",
    "CLD_DUMPED",
)


def launch_with_lifeline(
    command: list[str],
    *,
    wait: bool,
    popen_kwargs: dict[str, Any] | None = None,
    termination_grace_period: float = 5.0,
) -> int | None:
    """Launch a supervised command through a supervisor with a lifeline FD.

    When ``wait`` is ``False``, this returns after the supervisor starts and
    reports that the command was launched. When ``wait`` is ``True``, this
    waits for the supervisor and returns its exit code.
    ``popen_kwargs`` must be JSON-serializable because the launch request is
    sent to the supervisor over a config pipe.
    """
    if os.name != "posix":
        raise RuntimeError("lifeline supervision requires POSIX FD inheritance")
    _validate_non_reaping_wait_supported()
    _validate_launch_request(command, popen_kwargs)
    _validate_termination_grace_period(termination_grace_period)
    # Lifeline pipe: SuperExec keeps the write end, supervisor watches the read end.
    lifeline_read_fd, lifeline_write_fd = os.pipe()
    # Config pipe: SuperExec sends token-bearing TaskExecutor command details
    # off-argv.
    config_read_fd, config_write_fd = os.pipe()
    # Status pipe: supervisor reports command launch success/failure before we return.
    status_read_fd, status_write_fd = os.pipe()
    supervisor: subprocess.Popen[bytes] | None = None

    try:
        # Keep command details out of supervisor argv so token-bearing launch
        # commands are not exposed through process listings.
        supervisor_command = [
            sys.executable,
            "-m",
            "flwr.supercore.superexec.taskexecutor_supervisor",
            "--lifeline-fd",
            str(lifeline_read_fd),
            "--config-fd",
            str(config_read_fd),
            "--status-fd",
            str(status_write_fd),
            "--termination-grace-period",
            str(termination_grace_period),
        ]
        supervisor = subprocess.Popen(  # pylint: disable=consider-using-with
            supervisor_command,
            close_fds=True,
            env=_supervisor_env(),
            pass_fds=(lifeline_read_fd, config_read_fd, status_write_fd),
        )
        _close_fd(lifeline_read_fd)
        lifeline_read_fd = -1
        _close_fd(config_read_fd)
        config_read_fd = -1
        _close_fd(status_write_fd)
        status_write_fd = -1

        _write_config(config_write_fd, command, popen_kwargs or {})
        _close_fd(config_write_fd)
        config_write_fd = -1
        _check_launch_status(status_read_fd)
        _close_fd(status_read_fd)
        status_read_fd = -1

        if wait:
            # Keep the lifeline open while SuperExec waits so normal process exit
            # is not misinterpreted as parent death by the supervisor.
            try:
                return supervisor.wait()
            finally:
                _close_fd(lifeline_write_fd)
                lifeline_write_fd = -1

        _start_supervisor_reaper(supervisor, lifeline_write_fd)
        lifeline_write_fd = -1
        return None
    except Exception:
        if supervisor is not None and supervisor.poll() is None:
            _stop_supervisor_process(supervisor)
        raise
    finally:
        _close_fd(lifeline_read_fd)
        _close_fd(lifeline_write_fd)
        _close_fd(config_read_fd)
        _close_fd(config_write_fd)
        _close_fd(status_read_fd)
        _close_fd(status_write_fd)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the TaskExecutor supervisor entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run a supervised Flower TaskExecutor command."
    )
    parser.add_argument(
        "--lifeline-fd",
        type=int,
        required=True,
        help=(
            "Read end of the parent-owned lifeline pipe; EOF means SuperExec exited."
        ),
    )
    parser.add_argument(
        "--config-fd",
        type=int,
        required=True,
        help="Read end of the config pipe carrying command and subprocess kwargs.",
    )
    parser.add_argument(
        "--status-fd",
        type=int,
        required=True,
        help=(
            "Write end of the status pipe reporting command launch success or "
            "failure."
        ),
    )
    parser.add_argument(
        "--termination-grace-period",
        type=float,
        default=5.0,
        help=(
            "Seconds between SIGTERM and SIGKILL when cleaning the supervised "
            "process group."
        ),
    )
    args = parser.parse_args(argv)
    config_fd = args.config_fd
    lifeline_fd = args.lifeline_fd
    status_fd = args.status_fd

    try:
        _validate_termination_grace_period(args.termination_grace_period)
        os.set_inheritable(lifeline_fd, False)
        config = _read_config(config_fd)
        _close_fd(config_fd)
        config_fd = -1
        command = config["command"]
        popen_kwargs = config["popen_kwargs"]
        _validate_launch_request(command, popen_kwargs)
        return _run_supervised_process(
            command,
            lifeline_fd=lifeline_fd,
            popen_kwargs=popen_kwargs,
            termination_grace_period=args.termination_grace_period,
            status_fd=status_fd,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _write_launch_status(status_fd, ok=False, error=str(exc))
        print(f"Failed to supervise command: {exc}", file=sys.stderr)
        return 2
    finally:
        _close_fd(config_fd)
        _close_fd(lifeline_fd)
        _close_fd(status_fd)


def _run_supervised_process(
    command: list[str],
    *,
    lifeline_fd: int,
    popen_kwargs: dict[str, Any],
    termination_grace_period: float,
    status_fd: int | None = None,
) -> int:
    """Launch and supervise the command in its own process group.

    The loop uses a persistent selector for the lifeline FD and checks process exit
    without reaping when possible. That keeps process-group cleanup tied to the original
    group leader until the final wait.
    """
    _validate_non_reaping_wait_supported()
    supervised_process = subprocess.Popen(  # pylint: disable=consider-using-with
        command,
        **popen_kwargs,
        start_new_session=True,
        close_fds=True,
    )
    if status_fd is not None:
        _write_launch_status(status_fd, ok=True)
        _close_fd(status_fd)
    process_group_id = supervised_process.pid
    returncode: int | None = None
    lifeline_selector = selectors.DefaultSelector()
    try:
        lifeline_selector.register(lifeline_fd, selectors.EVENT_READ)
        while True:
            returncode = _peek_process_returncode(supervised_process)
            if returncode is not None:
                return returncode
            if _lifeline_closed(lifeline_selector, lifeline_fd):
                # EOF on the pipe means SuperExec exited or deliberately closed its
                # control end; cleanup is enforced outside any TaskExecutor PID
                # namespace.
                _terminate_process_group(
                    process_group_id,
                    supervised_process,
                    termination_grace_period,
                )
                returncode = supervised_process.wait()
                return returncode
            time.sleep(_POLL_INTERVAL)
    finally:
        # The wrapper/TaskExecutor can leave same-process-group children after the
        # leader exits; cleanup the whole group even on normal return.
        _terminate_process_group(
            process_group_id,
            supervised_process,
            termination_grace_period,
        )
        if returncode is not None and supervised_process.returncode is None:
            supervised_process.wait()
        lifeline_selector.close()


def _lifeline_closed(
    lifeline_selector: selectors.BaseSelector,
    lifeline_fd: int,
) -> bool:
    """Return True if the lifeline FD has reached EOF."""
    events = lifeline_selector.select(timeout=0)
    if not events:
        return False
    return os.read(lifeline_fd, 1) == b""


def _peek_process_returncode(supervised_process: subprocess.Popen[bytes]) -> int | None:
    """Return process exit status without reaping the process when possible.

    The supervisor must clean up the process group before it reaps the group leader;
    otherwise, a reused PID/PGID could receive cleanup signals meant for the supervised
    TaskExecutor group.
    """
    if supervised_process.returncode is not None:
        return supervised_process.returncode
    waitid_fn = _non_reaping_waitid()
    try:
        wait_result = waitid_fn(  # pylint: disable=not-callable
            os.P_PID,
            supervised_process.pid,
            os.WEXITED | os.WNOHANG | os.WNOWAIT,
        )
    except ChildProcessError as exc:
        raise RuntimeError(
            "unable to inspect supervised process status without reaping"
        ) from exc
    returncode: int | None = None
    if wait_result is None:
        return returncode
    if wait_result.si_code == os.CLD_EXITED:
        returncode = int(wait_result.si_status)
    elif wait_result.si_code in (os.CLD_KILLED, os.CLD_DUMPED):
        returncode = -int(wait_result.si_status)
    return returncode


def _terminate_process_group(
    process_group_id: int,
    supervised_process: subprocess.Popen[bytes],
    grace_period: float,
) -> None:
    """Terminate the supervised process group, escalating to SIGKILL if necessary."""
    if not _process_group_exists(process_group_id):
        return

    _send_signal_to_process_group(process_group_id, signal.SIGTERM)
    deadline = time.monotonic() + grace_period
    while time.monotonic() < deadline:
        # If the group leader has exited, reap it so the process group can
        # disappear once any remaining children have handled SIGTERM.
        if (
            _peek_process_returncode(supervised_process) is not None
            and supervised_process.returncode is None
        ):
            supervised_process.wait()
        if not _process_group_exists(process_group_id):
            return
        time.sleep(_POLL_INTERVAL)
    if _process_group_exists(process_group_id):
        _send_signal_to_process_group(process_group_id, signal.SIGKILL)


def _send_signal_to_process_group(process_group_id: int, sig: signal.Signals) -> None:
    """Send a signal to a process group, ignoring already-exited groups."""
    try:
        os.killpg(process_group_id, sig)
    except ProcessLookupError:
        pass
    except PermissionError as exc:
        raise RuntimeError(
            f"permission denied while signaling process group {process_group_id}"
        ) from exc


def _process_group_exists(process_group_id: int) -> bool:
    """Return True if a process group still exists."""
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError as exc:
        raise RuntimeError(
            f"permission denied while checking process group {process_group_id}"
        ) from exc
    return True


def _validate_non_reaping_wait_supported() -> None:
    """Raise if this platform cannot check child status without reaping.

    Falling back to ``Popen.poll``/``wait`` would reap the process-group leader before
    group cleanup and could let PID/PGID reuse redirect cleanup signals.
    """
    _non_reaping_waitid()


def _non_reaping_waitid() -> Callable[[int, int, int], Any]:
    """Return ``os.waitid`` only when the required non-reaping flags exist."""
    waitid = getattr(os, "waitid", None)
    missing = [
        attr for attr in _NON_REAPING_WAIT_ATTRS if getattr(os, attr, None) is None
    ]
    if not callable(waitid) or missing:
        missing_details = ["waitid"] if not callable(waitid) else []
        missing_details.extend(missing)
        raise RuntimeError(
            "lifeline supervision requires os.waitid with WNOWAIT support"
            f" ({', '.join(missing_details)} unavailable)"
        )
    return cast(Callable[[int, int, int], Any], waitid)


def _stop_supervisor_process(supervisor: subprocess.Popen[bytes]) -> None:
    """Terminate and reap a supervisor subprocess after failed setup."""
    supervisor.terminate()
    try:
        supervisor.wait(timeout=_SUPERVISOR_STOP_TIMEOUT)
    except subprocess.TimeoutExpired:
        supervisor.kill()
        supervisor.wait()


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
    if not all(isinstance(key, str) for key in popen_kwargs):
        raise TypeError("popen_kwargs keys must be strings")
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
    except (TypeError, ValueError) as exc:
        raise TypeError("popen_kwargs must be JSON serializable") from exc


def _validate_termination_grace_period(termination_grace_period: float) -> None:
    """Validate the process-group termination grace period."""
    if not math.isfinite(termination_grace_period) or termination_grace_period < 0:
        raise ValueError("termination_grace_period must be a finite non-negative value")


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


def _write_launch_status(
    status_fd: int,
    *,
    ok: bool,
    error: str | None = None,
) -> None:
    """Write command launch status back to the parent SuperExec process."""
    if status_fd < 0:
        return
    status: dict[str, object] = {"ok": ok}
    if error is not None:
        status["error"] = error
    status_bytes = json.dumps(status, separators=(",", ":")).encode("utf-8")
    try:
        while status_bytes:
            bytes_written = os.write(status_fd, status_bytes)
            status_bytes = status_bytes[bytes_written:]
    except OSError:
        return


def _check_launch_status(status_read_fd: int) -> None:
    """Raise if the supervisor failed before launching the command."""
    try:
        status = _read_json_fd(status_read_fd)
    except json.JSONDecodeError as exc:
        raise RuntimeError("supervisor exited before reporting launch status") from exc
    if not isinstance(status, dict):
        raise RuntimeError("supervisor returned an invalid launch status")
    if status.get("ok") is True:
        return
    error = status.get("error")
    if not isinstance(error, str) or not error:
        error = "unknown supervised process launch failure"
    raise RuntimeError(f"supervisor failed to launch command: {error}")


def _read_config(config_read_fd: int) -> dict[str, Any]:
    """Read launch config from the parent config pipe."""
    config = _read_json_fd(config_read_fd)
    if not isinstance(config, dict):
        raise TypeError("supervisor config must be a dictionary")
    return config


def _read_json_fd(fd: int) -> Any:
    """Read a JSON payload from an FD until EOF."""
    chunks = []
    while True:
        chunk = os.read(fd, _READ_SIZE)
        if not chunk:
            break
        chunks.append(chunk)
    return json.loads(b"".join(chunks).decode("utf-8"))


def _start_supervisor_reaper(
    supervisor: subprocess.Popen[bytes],
    lifeline_write_fd: int,
) -> None:
    """Close the lifeline control FD after the supervisor exits."""

    def reap() -> None:
        # For non-waiting launches, SuperExec must not leak the lifeline write FD
        # forever. The daemon reaper owns that FD once launch_with_lifeline returns.
        try:
            supervisor.wait()
        finally:
            # Closing this FD after supervisor exit also releases the descriptor if
            # the supervised process exits normally before SuperExec shuts down.
            _close_fd(lifeline_write_fd)

    # The thread is daemonized because SuperExec shutdown should still close the
    # process-owned FD and produce EOF for the supervisor even if the reaper is alive.
    threading.Thread(target=reap, daemon=True).start()


def _close_fd(fd: int) -> None:
    """Close an FD if it is open."""
    if fd < 0:
        return
    try:
        os.close(fd)
    except OSError:
        pass


if __name__ == "__main__":
    sys.exit(main())
