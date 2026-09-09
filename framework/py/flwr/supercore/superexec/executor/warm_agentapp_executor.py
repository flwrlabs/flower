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
"""Kubernetes exec handoff for one-task warm AgentApp executors."""


import time

from flwr.supercore.constant import (
    TASK_TYPE_TO_APPIO_API_ADDRESS_ARG,
    TASK_TYPE_TO_COMMAND,
)

from .types import ExecutionSpec

_TOKEN_STDIN_ACKNOWLEDGEMENT = "FLWR_AGENTAPP_TOKEN_ACCEPTED"


class WarmAgentAppUnavailable(RuntimeError):
    """Raised before a task token is sent to a warm AgentApp executor Pod."""


class KubernetesWarmAgentAppDispatch:
    """Interact with one Kubernetes exec stream without logging task authority."""

    def __init__(self, response: object) -> None:
        self._response = response

    def send_token(self, token: str) -> None:
        """Send one token over stdin without retaining it in Pod metadata."""
        write_stdin = getattr(self._response, "write_stdin", None)
        if not callable(write_stdin):
            raise WarmAgentAppUnavailable(
                "Kubernetes exec stream does not support standard input."
            )
        write_stdin(f"{token}\n")

    def wait_for_acceptance(self, timeout: float) -> bool:
        """Return whether the task child acknowledged consuming the token."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            stdout = self._read_stdout()
            self._read_stderr()
            self._read_all()
            if _TOKEN_STDIN_ACKNOWLEDGEMENT in stdout:
                return True
            if not self._is_open():
                return False
            self._update(min(0.5, deadline - time.monotonic()))
        return False

    def wait_for_close(self) -> None:
        """Wait for the one-task child to exit or the exec stream to close."""
        while self._is_open():
            self._update(1.0)
            self._read_stdout()
            self._read_stderr()
            self._read_all()

    def close(self) -> None:
        """Close the Kubernetes exec stream best-effort."""
        close = getattr(self._response, "close", None)
        if callable(close):
            close()

    def _is_open(self) -> bool:
        is_open = getattr(self._response, "is_open", None)
        return bool(is_open()) if callable(is_open) else False

    def _update(self, timeout: float) -> None:
        update = getattr(self._response, "update", None)
        if callable(update):
            update(timeout=max(timeout, 0.0))

    def _read_stdout(self) -> str:
        peek_stdout = getattr(self._response, "peek_stdout", None)
        read_stdout = getattr(self._response, "read_stdout", None)
        if not callable(read_stdout) or (callable(peek_stdout) and not peek_stdout()):
            return ""
        stdout = read_stdout()
        return stdout if isinstance(stdout, str) else ""

    def _read_stderr(self) -> None:
        peek_stderr = getattr(self._response, "peek_stderr", None)
        read_stderr = getattr(self._response, "read_stderr", None)
        if callable(read_stderr) and (not callable(peek_stderr) or peek_stderr()):
            read_stderr()

    def _read_all(self) -> None:
        read_all = getattr(self._response, "read_all", None)
        if callable(read_all):
            read_all()


def warm_agentapp_command(
    spec: ExecutionSpec, runtime_root_certificates: str | None
) -> list[str]:
    """Build a one-task child command that receives authority on standard input."""
    command = [
        TASK_TYPE_TO_COMMAND[spec.task_type],
        TASK_TYPE_TO_APPIO_API_ADDRESS_ARG[spec.task_type],
        spec.runtime_api_address,
        "--token-stdin",
    ]
    if spec.insecure:
        command.append("--insecure")
    elif runtime_root_certificates is not None:
        raise WarmAgentAppUnavailable(
            "Warm executor dispatch cannot safely deliver Runtime API certificates."
        )
    if spec.runtime_dependency_install:
        command.append("--allow-runtime-dependency-installation")
    return command
