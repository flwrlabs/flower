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
"""Single-use prestarted worker for warm Model TaskExecutor Pods."""

from __future__ import annotations

import argparse
import signal
import socket
import sys
import threading
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from logging import ERROR
from pathlib import Path
from types import FrameType
from typing import Any, Protocol

from flwr.common.args import add_args_flwr_app_common, try_obtain_flwr_app_token
from flwr.common.constant import FLWR_TASK_TOKEN_STDIN_ACKNOWLEDGEMENT
from flwr.supercore import log
from flwr.supercore import model_worker_protocol as protocol
from flwr.supercore.typing import JSONObject
from flwr.supercore.warm_executor_constants import (
    WARM_EXECUTOR_BUSY_FILE,
    WARM_EXECUTOR_READY_FILE,
    WARM_MODEL_EXECUTOR_SOCKET,
)

# Bound each socket write so a lost reader cannot stall accepted-task cleanup.
_OUTPUT_SEND_TIMEOUT_SECONDS = 0.1


class _RunModelOnce(Protocol):
    """Execute one Model task after task-scoped state is initialized."""

    def __call__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        runtime_api_address: str,
        token: str,
        insecure: bool,
        root_certificates: bytes | None,
        on_started: Callable[[], None],
        /,
    ) -> int:
        """Execute one Model task and return its process exit code."""


@dataclass(frozen=True)
class ModelInvocation:
    """Describe one token-scoped invocation for the prestarted Model worker."""

    token: str
    runtime_api_address: str
    insecure: bool
    root_certificates_path: str | None

    @classmethod
    def from_payload(cls, payload: object) -> ModelInvocation:
        """Validate and construct one invocation request."""
        if not isinstance(payload, dict):
            raise ValueError("Model invocation must be a JSON object.")
        if set(payload) != {
            "token",
            "runtime_api_address",
            "insecure",
            "root_certificates_path",
        }:
            raise ValueError("Model invocation contains unexpected fields.")

        token = _required_string(payload, "token")
        runtime_api_address = _required_string(payload, "runtime_api_address")
        insecure = payload["insecure"]
        root_certificates_path = payload["root_certificates_path"]
        if not isinstance(insecure, bool):
            raise ValueError("Model invocation field 'insecure' must be bool.")
        if root_certificates_path is not None and not isinstance(
            root_certificates_path, str
        ):
            raise ValueError(
                "Model invocation field 'root_certificates_path' must be string "
                "or null."
            )
        if isinstance(root_certificates_path, str) and not root_certificates_path:
            raise ValueError(
                "Model invocation field 'root_certificates_path' must not be empty."
            )
        if insecure and root_certificates_path is not None:
            raise ValueError(
                "Model invocation cannot combine insecure transport with root "
                "certificates."
            )
        return cls(
            token=token,
            runtime_api_address=runtime_api_address,
            insecure=insecure,
            root_certificates_path=root_certificates_path,
        )


def serve_prestarted_model_worker(
    socket_path: Path = Path(WARM_MODEL_EXECUTOR_SOCKET),
    ready_file: Path = Path(WARM_EXECUTOR_READY_FILE),
    busy_file: Path = Path(WARM_EXECUTOR_BUSY_FILE),
) -> int:
    """Preload the Model task path, then serve exactly one invocation."""
    # Keep the exec-side dispatcher lightweight. Only the resident process pays
    # the Model/provider import cost, and it does so before reporting readiness.
    from .task_process.model.run_model import (  # pylint: disable=import-outside-toplevel
        run_model_once,
    )

    _register_idle_signal_handlers()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    ready_file.unlink(missing_ok=True)
    busy_file.unlink(missing_ok=True)

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(socket_path))
            socket_path.chmod(0o600)
            server.listen(1)
            return _serve_ready_worker(server, ready_file, busy_file, run_model_once)
    finally:
        ready_file.unlink(missing_ok=True)
        busy_file.unlink(missing_ok=True)
        socket_path.unlink(missing_ok=True)


def _serve_ready_worker(
    server: socket.socket,
    ready_file: Path,
    busy_file: Path,
    run_once: _RunModelOnce,
) -> int:
    """Publish readiness, then make one accepted connection exclusively busy."""
    ready_file.touch()
    connection, _ = server.accept()
    busy_file.touch()
    ready_file.unlink(missing_ok=True)
    try:
        with connection:
            return _serve_connection(connection, run_once)
    finally:
        busy_file.unlink(missing_ok=True)


def dispatch_prestarted_model(  # pylint: disable=too-many-return-statements
    invocation: ModelInvocation,
    socket_path: Path = Path(WARM_MODEL_EXECUTOR_SOCKET),
) -> int:
    """Relay one exec-delivered invocation to the prestarted worker."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(socket_path))
            with connection.makefile("rwb") as channel:
                protocol.send_message(channel, asdict(invocation))
                accepted = False
                while True:
                    response = protocol.read_message(channel)
                    event_name = response.get("event")
                    if event_name == "accepted":
                        if accepted or set(response) != {"event"}:
                            return 1
                        accepted = True
                        print(FLWR_TASK_TOKEN_STDIN_ACKNOWLEDGEMENT, flush=True)
                    elif event_name == "rejected":
                        reason = response.get("reason")
                        if (
                            accepted
                            or set(response) != {"event", "reason"}
                            or not isinstance(reason, str)
                        ):
                            return 1
                        print(
                            reason,
                            file=sys.stderr,
                        )
                        return 1
                    elif event_name == "output":
                        stream = response.get("stream")
                        output = response.get("data")
                        if (
                            not accepted
                            or set(response) != {"event", "stream", "data"}
                            or stream not in {"stdout", "stderr"}
                            or not isinstance(output, str)
                        ):
                            return 1
                        output_stream = sys.stdout if stream == "stdout" else sys.stderr
                        try:
                            output_stream.write(output)
                            output_stream.flush()
                        except (OSError, ValueError):
                            # Output remains best effort after task acceptance.
                            pass
                    elif event_name == "finished":
                        returncode = response.get("returncode")
                        return (
                            returncode
                            if accepted
                            and set(response) == {"event", "returncode"}
                            and isinstance(returncode, int)
                            and not isinstance(returncode, bool)
                            else 1
                        )
                    else:
                        return 1
    except (OSError, ValueError) as err:
        print(f"Prestarted Model dispatch failed: {err}", file=sys.stderr)
        return 1


def _serve_connection(connection: socket.socket, run_once: _RunModelOnce) -> int:
    """Consume one invocation after the worker has become exclusively busy."""
    channel = connection.makefile("rwb")
    try:
        return _serve_channel(
            channel,
            run_once,
            prepare_output=lambda: connection.settimeout(_OUTPUT_SEND_TIMEOUT_SECONDS),
        )
    finally:
        try:
            channel.close()
        except OSError:
            # An accepted task must not fail because its exec-side reader left.
            pass


def _send_rejection(channel: protocol.MessageChannel, reason: str) -> None:
    """Send a rejection when the dispatcher is still reachable."""
    try:
        protocol.send_message(channel, {"event": "rejected", "reason": reason})
    except (OSError, ValueError):
        pass


def _serve_channel(
    channel: protocol.MessageChannel,
    run_once: _RunModelOnce,
    prepare_output: Callable[[], None] | None = None,
) -> int:
    """Consume one invocation over a buffered JSON-lines channel."""
    try:
        invocation = ModelInvocation.from_payload(protocol.read_message(channel))
    except ValueError as err:
        _send_rejection(channel, str(err))
        return 1

    sender: protocol.ProtocolOutputSender | None = None
    accepted = False
    output_context = ExitStack()

    def accept_invocation() -> None:
        nonlocal accepted, sender
        if accepted:
            raise RuntimeError("Model invocation was accepted more than once.")
        accepted = True
        try:
            protocol.send_message(channel, {"event": "accepted"})
        except Exception:  # pylint: disable=broad-exception-caught
            # Authority has reached task-scoped state. Execute instead of
            # risking duplicate processing through any fallback path.
            pass
        if prepare_output is not None:
            try:
                prepare_output()
            except Exception:  # pylint: disable=broad-exception-caught
                # Output delivery remains best effort after acceptance.
                pass
        sender = protocol.ProtocolOutputSender(channel)
        try:
            output_context.enter_context(
                protocol.relay_task_output(sender, invocation.token)
            )
        except Exception:  # pylint: disable=broad-exception-caught
            # Task execution must not depend on output relay setup.
            pass

    returncode = 1
    try:
        certificates = (
            Path(invocation.root_certificates_path).read_bytes()
            if invocation.root_certificates_path is not None
            else None
        )
        returncode = run_once(
            invocation.runtime_api_address,
            invocation.token,
            invocation.insecure,
            certificates,
            accept_invocation,
        )
        if not accepted:
            returncode = 1
            raise RuntimeError("Model invocation was not accepted.")
    except SystemExit as err:
        returncode = _system_exit_returncode(err)
        raise
    except Exception as err:  # pylint: disable=broad-exception-caught
        log(ERROR, "Prestarted Model worker failed", exc_info=err)
        if not accepted:
            _send_rejection(channel, "Prestarted Model worker could not start task.")
    finally:
        output_context.close()
        if sender is not None:
            sender.close(returncode)
    return returncode


def _system_exit_returncode(err: SystemExit) -> int:
    """Convert a SystemExit value to a protocol return code."""
    if err.code is None:
        return 0
    if isinstance(err.code, int):
        return int(err.code)
    return 1


def _register_idle_signal_handlers() -> None:
    """Let an idle PID 1 worker unwind marker and socket cleanup on signals."""
    from flwr.supercore.exit.signal_handler import (  # pylint: disable=import-outside-toplevel
        SIGNAL_TO_EXIT_CODE,
    )

    default_handlers: dict[int, Any] = {}
    is_exiting = False
    lock = threading.Lock()

    def graceful_exit_handler(_signalnum: int, _frame: FrameType | None) -> None:
        nonlocal is_exiting
        with lock:
            if is_exiting:
                return
            is_exiting = True
        for sig, default_handler in default_handlers.items():
            signal.signal(sig, default_handler)
        raise SystemExit(0)

    for sig in SIGNAL_TO_EXIT_CODE:
        default_handlers[sig] = signal.signal(sig, graceful_exit_handler)


def _required_string(payload: JSONObject, name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Model invocation field '{name}' must be a non-empty string.")
    return value


def _parse_args() -> argparse.ArgumentParser:
    """Build the internal prestarted Model worker argument parser."""
    parser = argparse.ArgumentParser(description="Run a prestarted Model worker")
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("serve")
    dispatch = modes.add_parser("dispatch")
    dispatch.add_argument("--runtime-api-address", required=True)
    add_args_flwr_app_common(dispatch, include_token_stdin=True)
    return parser


def main() -> None:
    """Run the resident worker or its exec-side dispatcher."""
    args = _parse_args().parse_args()
    if args.mode == "serve":
        raise SystemExit(serve_prestarted_model_worker())
    invocation = ModelInvocation(
        token=try_obtain_flwr_app_token(args),
        runtime_api_address=args.runtime_api_address,
        insecure=args.insecure,
        root_certificates_path=args.root_certificates,
    )
    raise SystemExit(dispatch_prestarted_model(invocation))


if __name__ == "__main__":
    main()
