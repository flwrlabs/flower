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
"""Helpers for wrapping SuperExec app launches in an app sandbox."""


import os
import shutil
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Literal

SandboxMode = Literal["disabled", "nsjail"]

_DEFAULT_NSJAIL_BINARY = "nsjail"
_DEFAULT_NSJAIL_CONFIG = "nsjail-flower-gpu-container.cfg"
_SANDBOX_CONFIG_PACKAGE = "flwr.supercore.superexec.sandbox.config"


@dataclass(frozen=True)
class SandboxConfig:
    """Resolved SuperExec app sandbox configuration."""

    mode: SandboxMode = "disabled"
    nsjail_binary: str = _DEFAULT_NSJAIL_BINARY
    nsjail_config_path: str | None = None

    @property
    def enabled(self) -> bool:
        """Return whether app launches should be sandboxed."""
        return self.mode != "disabled"

    @property
    def include_parent_pid(self) -> bool:
        """Return whether app commands should include parent PID monitoring."""
        return not self.enabled

    def wrap_command(self, command: list[str]) -> list[str]:
        """Wrap an app launch command in the configured sandbox."""
        if self.mode == "disabled":
            return command
        if self.mode == "nsjail" and self.nsjail_config_path is not None:
            resolved_command = _resolve_app_command(command)
            return [
                self.nsjail_binary,
                "--config",
                self.nsjail_config_path,
                "--",
                *resolved_command,
            ]
        raise ValueError(f"Unsupported SuperExec sandbox mode: {self.mode}")


def resolve_sandbox_config(
    mode: str | None = None,
    nsjail_config_path: str | None = None,
    nsjail_binary: str | None = None,
) -> SandboxConfig:
    """Resolve and validate SuperExec sandbox settings.

    Explicit nsjail requests fail closed. Disabled mode ignores unused nsjail
    settings to preserve existing SuperExec behavior.
    """
    resolved_mode = mode or os.getenv("FLWR_SUPEREXEC_SANDBOX", "disabled")
    resolved_config = nsjail_config_path or os.getenv("FLWR_SUPEREXEC_SANDBOX_CONFIG")
    resolved_binary = (
        nsjail_binary
        or os.getenv("FLWR_SUPEREXEC_NSJAIL_BINARY")
        or _DEFAULT_NSJAIL_BINARY
    )

    if resolved_mode not in {"disabled", "nsjail"}:
        raise ValueError(
            "Unsupported SuperExec sandbox mode "
            f"'{resolved_mode}'. Expected one of: disabled, nsjail."
        )

    if resolved_mode == "disabled":
        return SandboxConfig(mode="disabled")

    if not resolved_binary.strip():
        raise ValueError("SuperExec nsjail binary must not be empty.")

    executable = _resolve_executable(resolved_binary)
    config_path = _resolve_config_path(resolved_config)
    return SandboxConfig(
        mode="nsjail",
        nsjail_binary=executable,
        nsjail_config_path=config_path,
    )


def _resolve_executable(binary: str) -> str:
    """Resolve an nsjail binary name or path to an executable."""
    expanded = Path(binary).expanduser()
    if expanded.is_absolute() or os.sep in binary:
        if not expanded.is_file():
            raise ValueError(f"SuperExec nsjail binary not found: {expanded}")
        if not os.access(expanded, os.X_OK):
            raise ValueError(f"SuperExec nsjail binary is not executable: {expanded}")
        return str(expanded)

    resolved = shutil.which(binary)
    if resolved is None:
        raise ValueError(f"SuperExec nsjail binary not found in PATH: {binary}")
    return resolved


def _resolve_config_path(config_path: str | None) -> str:
    """Resolve an explicit or packaged nsjail config path."""
    if config_path:
        resolved = Path(config_path).expanduser()
    else:
        resolved = Path(
            str(files(_SANDBOX_CONFIG_PACKAGE).joinpath(_DEFAULT_NSJAIL_CONFIG))
        )

    if not resolved.is_file():
        raise ValueError(f"SuperExec nsjail config not found: {resolved}")
    if not os.access(resolved, os.R_OK):
        raise ValueError(f"SuperExec nsjail config is not readable: {resolved}")
    return str(resolved)


def _resolve_app_command(command: list[str]) -> list[str]:
    """Resolve app executable to an absolute path for nsjail execve."""
    if not command:
        raise ValueError("SuperExec app command must not be empty.")

    executable = command[0]
    expanded = Path(executable).expanduser()
    if expanded.is_absolute() or os.sep in executable:
        return [str(expanded), *command[1:]]

    resolved = shutil.which(executable)
    if resolved is None:
        raise ValueError(
            f"SuperExec app executable not found in PATH for nsjail: {executable}"
        )
    return [resolved, *command[1:]]
