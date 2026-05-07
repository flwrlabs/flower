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
"""Simple base Flower SuperExec plugin for app processes."""


import os
import shlex
import subprocess
import time
from collections.abc import Sequence
from logging import ERROR, INFO, WARNING
from pathlib import Path
from typing import Any

from flwr.common.logger import log

from .exec_plugin import ExecPlugin

_APP_LOG_DIR_ENV = "FLWR_SUPEREXEC_APP_LOG_DIR"
_STARTUP_CHECK_SECONDS_ENV = "FLWR_SUPEREXEC_APP_STARTUP_CHECK_SECONDS"
_LOG_TAIL_BYTES = 4096


class BaseExecPlugin(ExecPlugin):
    """Simple Flower SuperExec plugin for app processes.

    The plugin always selects the first candidate run ID.
    """

    # Placeholders to be defined in subclasses
    command = ""
    appio_api_address_arg = ""

    def select_run_id(self, candidate_run_ids: Sequence[int]) -> int | None:
        """Select a run ID to execute from a sequence of candidates."""
        if not candidate_run_ids:
            return None
        return candidate_run_ids[0]

    def launch_app(self, token: str, run_id: int) -> None:
        """Launch the application associated with a given run ID and token."""
        cmds = [self.command]
        if self.insecure:
            cmds.append("--insecure")
        elif self.root_certificates_path:
            cmds += ["--root-certificates", self.root_certificates_path]
        cmds += [self.appio_api_address_arg, self.appio_api_address]
        cmds += ["--token", token]
        if self.sandbox_config.include_parent_pid:
            cmds += ["--parent-pid", str(os.getpid())]
        if self.runtime_dependency_install:
            cmds += ["--allow-runtime-dependency-installation"]
        cmds = self.sandbox_config.wrap_command(cmds)
        popen_kwargs = self.get_popen_kwargs()
        log_path = self._prepare_launch_logging(
            popen_kwargs=popen_kwargs, run_id=run_id
        )
        log(
            INFO,
            "Launching app for run_id %d: %s%s",
            run_id,
            _format_command_for_log(cmds),
            f" (logs: {log_path})" if log_path else "",
        )
        log_handles = popen_kwargs.pop("_flwr_log_handles", [])
        # Launch the client app without waiting for it to complete.
        # Since we don't need to manage the process, we intentionally avoid using
        # a `with` statement. Suppress the pylint warning for it in this case.
        # pylint: disable-next=consider-using-with
        try:
            process = subprocess.Popen(cmds, **popen_kwargs)
        except OSError as exc:
            log(
                ERROR,
                "Failed to launch app for run_id %d: %s. Command: %s%s",
                run_id,
                exc,
                _format_command_for_log(cmds),
                f" Logs: {log_path}" if log_path else "",
            )
            raise
        finally:
            for handle in log_handles:
                handle.close()

        self._check_startup(
            process=process, run_id=run_id, cmds=cmds, log_path=log_path
        )

    def get_popen_kwargs(self) -> dict[str, Any]:
        """Return subprocess keyword arguments when launching app processes."""
        return {}

    def _prepare_launch_logging(
        self, popen_kwargs: dict[str, Any], run_id: int
    ) -> Path | None:
        """Attach stdout/stderr to a per-run log file if configured."""
        log_dir_value = os.getenv(_APP_LOG_DIR_ENV, "").strip()
        if not log_dir_value:
            return None

        try:
            log_dir = Path(log_dir_value).expanduser()
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{_safe_log_name(self.command)}-{run_id}.log"
            log_file = log_path.open("ab", buffering=0)
        except OSError as exc:
            log(
                WARNING,
                "Unable to open SuperExec app log file for run_id %d: %s",
                run_id,
                exc,
            )
            return None

        popen_kwargs["stdout"] = log_file
        popen_kwargs["stderr"] = subprocess.STDOUT
        popen_kwargs.setdefault("_flwr_log_handles", []).append(log_file)
        return log_path

    def _check_startup(
        self,
        process: subprocess.Popen[Any],
        run_id: int,
        cmds: list[str],
        log_path: Path | None,
    ) -> None:
        """Optionally detect app processes that exit immediately after launch."""
        startup_check_seconds = _startup_check_seconds()
        if startup_check_seconds <= 0:
            return

        time.sleep(startup_check_seconds)
        returncode = process.poll()
        if returncode is None:
            log(
                INFO,
                "App process for run_id %d is still running after %.2fs startup check.",
                run_id,
                startup_check_seconds,
            )
            return

        log(
            ERROR,
            "App process for run_id %d exited during startup check with code %s. "
            "Command: %s%s%s",
            run_id,
            returncode,
            _format_command_for_log(cmds),
            f" Logs: {log_path}" if log_path else "",
            _format_log_tail(log_path),
        )


def _startup_check_seconds() -> float:
    """Return configured app startup check duration."""
    raw_value = os.getenv(_STARTUP_CHECK_SECONDS_ENV, "0").strip()
    if not raw_value:
        return 0.0
    try:
        return max(float(raw_value), 0.0)
    except ValueError:
        log(
            WARNING,
            "Ignoring invalid %s value: %r",
            _STARTUP_CHECK_SECONDS_ENV,
            raw_value,
        )
        return 0.0


def _format_command_for_log(command: Sequence[str]) -> str:
    """Return shell-escaped command with sensitive token values redacted."""
    return shlex.join(_redact_command(command))


def _redact_command(command: Sequence[str]) -> list[str]:
    """Redact sensitive CLI argument values before logging commands."""
    redacted: list[str] = []
    redact_next = False
    for arg in command:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        redacted.append(arg)
        if arg == "--token":
            redact_next = True
    return redacted


def _safe_log_name(value: str) -> str:
    """Return a filesystem-safe log filename stem."""
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return safe or "app"


def _format_log_tail(log_path: Path | None) -> str:
    """Return a short log tail for immediate startup failures."""
    if log_path is None or not log_path.exists():
        return ""
    try:
        with log_path.open("rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            size = log_file.tell()
            log_file.seek(max(size - _LOG_TAIL_BYTES, 0))
            tail = log_file.read().decode("utf-8", errors="replace").strip()
    except OSError as exc:
        return f" Unable to read log tail: {exc}"
    if not tail:
        return ""
    return f" Last log output:\n{tail}"
