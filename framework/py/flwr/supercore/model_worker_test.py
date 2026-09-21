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
"""Tests for the single-use prestarted Model worker."""

# pylint: disable=protected-access

import socket
import sys
import threading
import time
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

from flwr.common.constant import FLWR_TASK_TOKEN_STDIN_ACKNOWLEDGEMENT

from . import model_worker, model_worker_protocol
from .model_worker import ModelInvocation

_RunOnce = Callable[[str, str, bool, bytes | None, Callable[[], None]], int]


def _invocation(**changes: Any) -> dict[str, Any]:
    payload = {
        "token": "task-token",
        "runtime_api_address": "runtime.example:9092",
        "insecure": True,
        "root_certificates_path": None,
    }
    payload.update(changes)
    return payload


def _exchange(
    run_once: _RunOnce,
    payload: dict[str, Any] | None = None,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Serve one socket-pair invocation and collect its protocol frames."""
    server, client = socket.socketpair()
    result: list[int] = []
    thread = threading.Thread(
        target=lambda: result.append(model_worker._serve_connection(server, run_once))
    )
    thread.start()
    frames: list[dict[str, Any]] = []
    try:
        with client.makefile("rwb") as channel:
            model_worker_protocol.send_message(channel, payload or _invocation())
            while True:
                frame = model_worker_protocol.read_message(channel)
                frames.append(frame)
                if frame.get("event") in {"finished", "rejected"}:
                    break
    finally:
        client.close()
        server.close()
        thread.join(timeout=2.0)
    assert not thread.is_alive()
    return result, frames


def test_protocol_rejects_invalid_messages() -> None:
    """Protocol messages should be typed, bounded, and fully written."""

    class ShortWriteChannel(BytesIO):
        """Accept only a few bytes from each write."""

        def write(self, data: Any, /) -> int:
            return super().write(data[:3])

    channel = ShortWriteChannel()
    model_worker_protocol.send_message(channel, {"value": "\ud800"})
    assert channel.getvalue() == b'{"value":"\\ud800"}\n'

    with pytest.raises(ValueError, match="valid JSON"):
        model_worker_protocol.read_message(BytesIO(b"{\n"))
    with pytest.raises(ValueError, match="non-empty string"):
        ModelInvocation.from_payload(_invocation(token=""))
    with pytest.raises(ValueError, match="unexpected fields"):
        ModelInvocation.from_payload(_invocation(extra=True))
    with pytest.raises(ValueError, match="too large"):
        model_worker_protocol.read_message(
            BytesIO(b"x" * (model_worker_protocol.MAX_PROTOCOL_MESSAGE_BYTES + 1))
        )


def test_worker_cleans_up_socket_and_markers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Worker files should be exclusive and removed after serving a task."""
    socket_path = tmp_path / "model.sock"
    ready_file = tmp_path / "ready"
    busy_file = tmp_path / "busy"
    connection = MagicMock()
    server = MagicMock()
    server.__enter__.return_value = server
    server.bind.side_effect = lambda _: socket_path.touch()
    unlink = Path.unlink

    def unlink_after_busy(path: Path, missing_ok: bool = False) -> None:
        if path == ready_file and path.exists():
            assert busy_file.exists()
        unlink(path, missing_ok=missing_ok)

    def accept() -> tuple[object, None]:
        assert ready_file.is_file()
        return connection, None

    def serve_connection(*_: object) -> int:
        assert busy_file.is_file()
        assert not ready_file.exists()
        return 0

    server.accept.side_effect = accept
    register_signals = Mock()
    monkeypatch.setattr(socket, "socket", Mock(return_value=server))
    monkeypatch.setattr(
        model_worker, "_register_idle_signal_handlers", register_signals
    )
    monkeypatch.setattr(model_worker, "_serve_connection", serve_connection)
    monkeypatch.setattr(Path, "unlink", unlink_after_busy)

    assert (
        model_worker.serve_prestarted_model_worker(socket_path, ready_file, busy_file)
        == 0
    )
    register_signals.assert_called_once_with()
    assert not any(path.exists() for path in (socket_path, ready_file, busy_file))


def test_dispatch_relays_accepted_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The dispatcher should acknowledge, then reproduce typed output."""
    connection = MagicMock()
    connection.__enter__.return_value = connection
    channel = MagicMock()
    channel.write.side_effect = len
    connection.makefile.return_value.__enter__.return_value = channel
    monkeypatch.setattr(socket, "socket", Mock(return_value=connection))
    monkeypatch.setattr(
        model_worker_protocol,
        "read_message",
        Mock(
            side_effect=[
                {"event": "accepted"},
                {"event": "output", "stream": "stdout", "data": "out\n"},
                {"event": "output", "stream": "stderr", "data": "err\n"},
                {"event": "finished", "returncode": 0},
            ]
        ),
    )

    assert (
        model_worker.dispatch_prestarted_model(
            ModelInvocation("task-token", "runtime.example:9092", True, None)
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.out == f"{FLWR_TASK_TOKEN_STDIN_ACKNOWLEDGEMENT}\nout\n"
    assert captured.err == "err\n"
    assert b"task-token" in channel.write.call_args.args[0]


def test_dispatch_rejects_output_before_acceptance(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Output without accepted task authority should not be relayed."""
    connection = MagicMock()
    connection.__enter__.return_value = connection
    channel = MagicMock()
    channel.write.side_effect = len
    connection.makefile.return_value.__enter__.return_value = channel
    monkeypatch.setattr(socket, "socket", Mock(return_value=connection))
    monkeypatch.setattr(
        model_worker_protocol,
        "read_message",
        Mock(return_value={"event": "output", "stream": "stdout", "data": "bad"}),
    )

    assert (
        model_worker.dispatch_prestarted_model(
            ModelInvocation("task-token", "runtime.example:9092", True, None)
        )
        == 1
    )
    assert capsys.readouterr().out == ""


def test_worker_relays_redacted_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Task output should be compatible, redacted, and carry fresh certificates."""
    token = "task-token-that-must-not-be-relayed"
    certificate_path = tmp_path / "runtime-ca.pem"
    certificate_path.write_bytes(b"test-ca")
    original_stdout = Mock(encoding="utf-8", buffer=Mock(raw=BytesIO()))
    original_stdout.fileno.return_value = 42
    monkeypatch.setattr(sys, "stdout", original_stdout)

    def run_once(
        runtime_api_address: str,
        invocation_token: str,
        insecure: bool,
        certificates: bytes | None,
        on_started: Callable[[], None],
    ) -> int:
        assert (runtime_api_address, insecure, certificates) == (
            "runtime.example:9092",
            False,
            b"test-ca",
        )
        on_started()
        assert sys.stdout.fileno() == 42
        binary_output: Any = sys.stdout.buffer
        binary_output.write(f"binary {invocation_token}\n".encode())
        binary_output.write1(f"write1 {invocation_token}\n".encode())
        binary_output.raw.write(f"raw {invocation_token}\n".encode())
        binary_output.writelines([f"binary lines {invocation_token}\n".encode()])
        sys.stdout.writelines(["text lines ", invocation_token, "\n"])
        midpoint = len(invocation_token) // 2
        sys.stdout.write(f"before {invocation_token[:midpoint]}")
        sys.stdout.write(f"{invocation_token[midpoint:]} after\n")
        sys.stderr.write("model error\n")
        sys.stderr.write(FLWR_TASK_TOKEN_STDIN_ACKNOWLEDGEMENT)
        return 0

    result, frames = _exchange(
        run_once,
        _invocation(
            token=token,
            insecure=False,
            root_certificates_path=str(certificate_path),
        ),
    )

    output = "".join(
        str(frame["data"]) for frame in frames if frame.get("event") == "output"
    )
    assert result == [0]
    assert frames[0] == {"event": "accepted"}
    assert frames[-1] == {"event": "finished", "returncode": 0}
    assert "binary [REDACTED]\n" in output
    assert "write1 [REDACTED]\nraw [REDACTED]\n" in output
    assert "binary lines [REDACTED]\ntext lines [REDACTED]\n" in output
    assert "before [REDACTED] after\n" in output
    assert "model error\n" in output
    assert token not in output
    assert FLWR_TASK_TOKEN_STDIN_ACKNOWLEDGEMENT not in output


def test_worker_rejects_failure_before_acceptance() -> None:
    """Returning without acceptance should reject and fail the worker."""
    result, frames = _exchange(Mock(return_value=0))

    assert result == [1]
    assert frames == [
        {
            "event": "rejected",
            "reason": "Prestarted Model worker could not start task.",
        }
    ]


def test_worker_reports_graceful_exit_after_acceptance() -> None:
    """A graceful resident exit should still send its terminal result."""
    channel = BytesIO(model_worker_protocol.encode_message(_invocation()))

    def stop(
        _runtime_api_address: str,
        _token: str,
        _insecure: bool,
        _certificates: bytes | None,
        on_started: Callable[[], None],
    ) -> int:
        on_started()
        raise SystemExit(0)

    with pytest.raises(SystemExit):
        model_worker._serve_channel(channel, stop)

    responses = BytesIO(channel.getvalue().partition(b"\n")[2])
    assert model_worker_protocol.read_message(responses) == {"event": "accepted"}
    assert model_worker_protocol.read_message(responses) == {
        "event": "finished",
        "returncode": 0,
    }


def test_disconnect_does_not_abort_accepted_task() -> None:
    """Losing the dispatcher must not change accepted-task execution."""
    server, client = socket.socketpair()
    started = threading.Event()
    release = threading.Event()

    def run_once(
        _runtime_api_address: str,
        _token: str,
        _insecure: bool,
        _certificates: bytes | None,
        on_started: Callable[[], None],
    ) -> int:
        on_started()
        started.set()
        release.wait(timeout=1.0)
        return 0

    run_once_mock = Mock(side_effect=run_once)
    result: list[int] = []
    thread = threading.Thread(
        target=lambda: result.append(
            model_worker._serve_connection(server, run_once_mock)
        )
    )
    thread.start()
    channel = client.makefile("rwb")
    model_worker_protocol.send_message(channel, _invocation())
    assert model_worker_protocol.read_message(channel) == {"event": "accepted"}
    assert started.wait(timeout=1.0)
    channel.close()
    client.close()
    release.set()
    thread.join(timeout=2.0)
    server.close()

    assert result == [0]
    run_once_mock.assert_called_once()


def test_saturated_output_channel_does_not_abort_task() -> None:
    """A slow or failed output write should preserve task completion."""

    class SlowChannel(BytesIO):
        """Delay every write after the synchronous acceptance frame."""

        def __init__(self, initial_bytes: bytes) -> None:
            super().__init__(initial_bytes)
            self.writes = 0

        def write(self, data: Any, /) -> int:
            self.writes += 1
            if self.writes > 1:
                time.sleep(0.02)
            if self.writes == 3:
                raise OSError("output write failed")
            return super().write(data)

        def flush(self) -> None:
            return

    channel = SlowChannel(model_worker_protocol.encode_message(_invocation()))

    def run_once(
        _runtime_api_address: str,
        _token: str,
        _insecure: bool,
        _certificates: bytes | None,
        on_started: Callable[[], None],
    ) -> int:
        on_started()
        sys.stdout.write("x" * 200_000)
        return 0

    started_at = time.monotonic()
    result = model_worker._serve_channel(channel, run_once)
    elapsed = time.monotonic() - started_at

    assert result == 0
    assert elapsed < 1.0
    assert b'{"event":"finished","returncode":0}\n' in channel.getvalue()
