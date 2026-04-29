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
"""Cooperative lifeline FD launcher for SuperExec app subprocesses."""

import os
import subprocess
import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AppLaunchResult:
    """Result for a non-waiting app launch."""

    pid: int


def launch_with_lifeline(
    command: list[str],
    *,
    wait: bool,
    popen_kwargs: dict[str, Any] | None = None,
) -> AppLaunchResult | int:
    """Launch an app command with a cooperative lifeline FD.

    The app receives ``--lifeline-fd <fd>`` and monitors EOF on that FD. SuperExec
    keeps the write end open until the app exits or SuperExec itself terminates.
    """
    if os.name != "posix":
        raise RuntimeError("lifeline FD launch requires POSIX FD inheritance")
    if not command or not all(isinstance(part, str) for part in command):
        raise ValueError("command must be a non-empty list of strings")

    app_lifeline_fd, superexec_lifeline_fd = os.pipe()
    proc: subprocess.Popen[bytes] | None = None
    try:
        kwargs = _prepare_popen_kwargs(popen_kwargs, app_lifeline_fd)
        proc = subprocess.Popen(  # pylint: disable=consider-using-with
            [*command, "--lifeline-fd", str(app_lifeline_fd)],
            **kwargs,
        )
        os.close(app_lifeline_fd)
        app_lifeline_fd = -1

        if wait:
            try:
                return proc.wait()
            finally:
                _close_fd(superexec_lifeline_fd)
                superexec_lifeline_fd = -1

        _start_lifeline_reaper(proc, superexec_lifeline_fd)
        superexec_lifeline_fd = -1
        return AppLaunchResult(pid=proc.pid)
    except Exception:
        if proc is not None and proc.poll() is None:
            proc.terminate()
        raise
    finally:
        _close_fd(app_lifeline_fd)
        _close_fd(superexec_lifeline_fd)


def _prepare_popen_kwargs(
    popen_kwargs: dict[str, Any] | None,
    app_lifeline_fd: int,
) -> dict[str, Any]:
    """Return Popen kwargs with explicit lifeline FD inheritance."""
    kwargs = dict(popen_kwargs or {})
    if kwargs.get("close_fds") is False:
        raise ValueError("lifeline FD launch requires close_fds=True")
    pass_fds = tuple(kwargs.pop("pass_fds", ()))
    # Only the app-side read FD should cross exec. Leaking SuperExec's write end
    # would keep the lifeline open after SuperExec exits.
    kwargs["close_fds"] = True
    kwargs["pass_fds"] = (*pass_fds, app_lifeline_fd)
    return kwargs


def _start_lifeline_reaper(
    proc: subprocess.Popen[bytes],
    superexec_lifeline_fd: int,
) -> None:
    """Close SuperExec's lifeline FD after the app process exits."""

    def reap() -> None:
        try:
            proc.wait()
        finally:
            _close_fd(superexec_lifeline_fd)

    threading.Thread(target=reap, daemon=True).start()


def _close_fd(fd: int) -> None:
    """Close an FD if it is open."""
    if fd < 0:
        return
    try:
        os.close(fd)
    except OSError:
        return
