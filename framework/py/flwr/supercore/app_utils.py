# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Utility functions for app processes."""


import os
import signal
import threading
import time

if os.name == "nt":
    from ctypes import windll  # type: ignore


def _pid_exists(pid: int) -> bool:
    """Check if a process with the given PID exists.

    This works on Unix-like systems and Windows.
    """
    # Use `ctypes` to check if the process exists on Windows
    if os.name == "nt":
        handle = windll.kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            windll.kernel32.CloseHandle(handle)
            return True
        return False
    # Use `os.kill` on Unix-like systems
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def start_parent_process_monitor(
    parent_pid: int,
) -> None:
    """Monitor the parent process and exit if it terminates."""

    def monitor() -> None:
        while True:
            time.sleep(0.2)
            if not _pid_exists(parent_pid):
                signal.raise_signal(signal.SIGINT)
                break

    threading.Thread(target=monitor, daemon=True).start()
