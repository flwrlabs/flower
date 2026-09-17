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
import json
import socket
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from logging import ERROR
from pathlib import Path
from typing import Any, Protocol

from flwr.common.args import add_args_flwr_app_common, try_obtain_flwr_app_token
from flwr.common.constant import FLWR_TASK_TOKEN_STDIN_ACKNOWLEDGEMENT
from flwr.supercore import log
from flwr.supercore.warm_executor_constants import (
    WARM_EXECUTOR_BUSY_FILE,
    WARM_EXECUTOR_READY_FILE,
    WARM_MODEL_EXECUTOR_SOCKET,
)

_MAX_PROTOCOL_MESSAGE_BYTES = 16_384
_RunModelOnce = Callable[[str, str, bool, bytes | None], int]


class _MessageChannel(Protocol):
    """Minimal buffered byte-stream interface used by the worker protocol."""

    def readline(self, size: int = -1, /) -> bytes:
        """Read at most one protocol line."""

    def write(self, data: bytes, /) -> int:
        """Write protocol bytes."""

    def flush(self) -> None:
        """Flush pending protocol bytes."""


@dataclass(frozen=True)
class ModelInvocation:
    """Describe one token-scoped invocation for the prestarted Model worker."""

    token: str
    runtime_api_address: str
    insecure: bool
    root_certificates_path: str | None

    @classmethod
    def from_json(cls, raw: str) -> ModelInvocation:
        """Parse and validate one invocation request."""
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as err:
            raise ValueError("Model invocation must be valid JSON.") from err

        return cls.from_payload(payload)

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
    ready_file.unlink(missing_ok=True)
    busy_file.touch()
    try:
        with connection:
            return _serve_connection(connection, run_once)
    finally:
        busy_file.unlink(missing_ok=True)


def dispatch_prestarted_model(
    invocation: ModelInvocation,
    socket_path: Path = Path(WARM_MODEL_EXECUTOR_SOCKET),
) -> int:
    """Relay one exec-delivered invocation to the prestarted worker."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(socket_path))
            with connection.makefile("rwb") as channel:
                _send_message(channel, asdict(invocation))
                accepted = False
                while True:
                    response = _read_message(channel)
                    event_name = response.get("event")
                    if event_name == "accepted":
                        accepted = True
                        print(FLWR_TASK_TOKEN_STDIN_ACKNOWLEDGEMENT, flush=True)
                    elif event_name == "rejected":
                        print(
                            str(response.get("reason", "Model invocation rejected.")),
                            file=sys.stderr,
                        )
                        return 1
                    elif event_name == "finished":
                        returncode = response.get("returncode")
                        return (
                            returncode
                            if accepted and isinstance(returncode, int)
                            else 1
                        )
                    else:
                        return 1
    except (OSError, ValueError) as err:
        print(f"Prestarted Model dispatch failed: {err}", file=sys.stderr)
        return 1


def _serve_connection(connection: socket.socket, run_once: _RunModelOnce) -> int:
    """Consume one invocation after the worker has become exclusively busy."""
    with connection.makefile("rwb") as channel:
        return _serve_channel(channel, run_once)


def _serve_channel(channel: _MessageChannel, run_once: _RunModelOnce) -> int:
    """Consume one invocation over a buffered JSON-lines channel."""
    try:
        invocation = ModelInvocation.from_payload(_read_message(channel))
    except ValueError as err:
        try:
            _send_message(channel, {"event": "rejected", "reason": str(err)})
        except OSError:
            pass
        return 1

    try:
        _send_message(channel, {"event": "accepted"})
    except OSError:
        # Authority has already reached the worker. Execute instead of risking
        # duplicate task processing through a fallback path.
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
        )
    except Exception as err:  # pylint: disable=broad-exception-caught
        log(ERROR, "Prestarted Model worker failed", exc_info=err)
    try:
        _send_message(channel, {"event": "finished", "returncode": returncode})
    except OSError:
        pass
    return returncode


def _send_message(channel: _MessageChannel, payload: dict[str, Any]) -> None:
    """Send one bounded newline-delimited JSON protocol message."""
    encoded = json.dumps(payload, separators=(",", ":")).encode() + b"\n"
    if len(encoded) > _MAX_PROTOCOL_MESSAGE_BYTES:
        raise ValueError("Model worker protocol message is too large.")
    channel.write(encoded)
    channel.flush()


def _read_message(channel: _MessageChannel) -> dict[str, Any]:
    """Read one bounded newline-delimited JSON protocol message."""
    encoded = channel.readline(_MAX_PROTOCOL_MESSAGE_BYTES + 1)
    if not encoded:
        raise ValueError("Model worker protocol connection closed.")
    if len(encoded) > _MAX_PROTOCOL_MESSAGE_BYTES or not encoded.endswith(b"\n"):
        raise ValueError("Model worker protocol message is too large.")
    raw = encoded[:-1]
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise ValueError("Model worker protocol message must be valid JSON.") from err
    if not isinstance(payload, dict):
        raise ValueError("Model worker protocol message must be a JSON object.")
    return payload


def _required_string(payload: dict[str, Any], name: str) -> str:
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
