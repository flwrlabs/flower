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

# pylint: disable=protected-access,too-many-locals

import multiprocessing
import os
import signal
import socket
import sys
import tempfile
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


def test_model_invocation_rejects_missing_and_unexpected_fields() -> None:
    """The worker protocol should accept only the expected typed payload."""
    with pytest.raises(ValueError, match="non-empty string"):
        ModelInvocation.from_json(
            '{"token":"","runtime_api_address":"runtime:9092",'
            '"insecure":true,"root_certificates_path":null}'
        )
    with pytest.raises(ValueError, match="unexpected fields"):
        ModelInvocation.from_json(
            '{"token":"token","runtime_api_address":"runtime:9092",'
            '"insecure":true,"root_certificates_path":null,"extra":true}'
        )


def test_protocol_preserves_coalesced_messages() -> None:
    """Buffered reads should preserve a completion sent with its acceptance."""
    channel = BytesIO(b'{"event":"accepted"}\n{"event":"finished","returncode":0}\n')

    assert model_worker_protocol.read_message(channel) == {"event": "accepted"}
    assert model_worker_protocol.read_message(channel) == {
        "event": "finished",
        "returncode": 0,
    }


def test_protocol_rejects_oversized_message() -> None:
    """The worker should never allocate an unbounded protocol message."""
    channel = BytesIO(b"x" * (model_worker_protocol.MAX_PROTOCOL_MESSAGE_BYTES + 1))

    with pytest.raises(ValueError, match="too large"):
        model_worker_protocol.read_message(channel)


@pytest.mark.parametrize("task_returncode", [0, 1])
def test_prestarted_worker_serves_one_invocation_and_cleans_up(
    tmp_path: Path,
    task_returncode: int,
) -> None:
    """The resident process should become busy, run once, and remove markers."""
    ready_file = tmp_path / "ready"
    busy_file = tmp_path / "busy"

    def run_once(
        runtime_api_address: str,
        token: str,
        insecure: bool,
        certificates: bytes | None,
        on_started: Callable[[], None],
    ) -> int:
        assert runtime_api_address == "runtime.example:9092"
        assert token == "task-token"
        assert insecure
        assert certificates is None
        assert busy_file.is_file()
        assert not ready_file.exists()
        on_started()
        return task_returncode

    run_model_once = Mock(side_effect=run_once)
    server_connection, client_connection = socket.socketpair()
    listener = Mock()
    accept_invocation = threading.Event()

    def accept() -> tuple[object, None]:
        accept_invocation.wait(timeout=2.0)
        return server_connection, None

    listener.accept.side_effect = accept
    result: list[int] = []
    thread = threading.Thread(
        target=lambda: result.append(
            model_worker._serve_ready_worker(  # pylint: disable=protected-access
                listener, ready_file, busy_file, run_model_once
            )
        )
    )
    thread.start()
    deadline = time.monotonic() + 2.0
    while not ready_file.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready_file.is_file()
    accept_invocation.set()

    with client_connection.makefile("rwb") as channel:
        model_worker_protocol.send_message(
            channel,
            {
                "token": "task-token",
                "runtime_api_address": "runtime.example:9092",
                "insecure": True,
                "root_certificates_path": None,
            },
        )
        assert model_worker_protocol.read_message(channel) == {"event": "accepted"}
        assert model_worker_protocol.read_message(channel) == {
            "event": "finished",
            "returncode": task_returncode,
        }
    thread.join(timeout=2.0)
    client_connection.close()

    assert result == [task_returncode]
    assert not thread.is_alive()
    call_args = run_model_once.call_args.args
    assert call_args[:4] == (
        "runtime.example:9092",
        "task-token",
        True,
        None,
    )
    assert callable(call_args[4])
    assert not ready_file.exists()
    assert not busy_file.exists()


def test_prestarted_worker_cleans_up_after_rejection(tmp_path: Path) -> None:
    """A malformed invocation should remove ready and busy markers."""
    ready_file = tmp_path / "ready"
    busy_file = tmp_path / "busy"
    server_connection, client_connection = socket.socketpair()
    listener = Mock()
    listener.accept.return_value = (server_connection, None)
    result: list[int] = []
    thread = threading.Thread(
        target=lambda: result.append(
            model_worker._serve_ready_worker(
                listener,
                ready_file,
                busy_file,
                Mock(),
            )
        )
    )
    thread.start()
    try:
        with client_connection.makefile("rwb") as channel:
            model_worker_protocol.send_message(channel, {"unexpected": True})
            response = model_worker_protocol.read_message(channel)
            assert response["event"] == "rejected"
    finally:
        client_connection.close()
        server_connection.close()
        thread.join(timeout=2.0)

    assert result == [1]
    assert not ready_file.exists()
    assert not busy_file.exists()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX signals")
def test_idle_worker_signal_cleans_up_markers_and_socket() -> None:
    """An idle PID 1 signal should unwind all worker filesystem state."""
    with tempfile.TemporaryDirectory(prefix="flwr-model-", dir="/tmp") as directory:
        worker_directory = Path(directory)
        socket_path = worker_directory / "model.sock"
        ready_file = worker_directory / "ready"
        busy_file = worker_directory / "busy"
        probe_path = worker_directory / "probe.sock"
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                probe.bind(str(probe_path))
        except OSError:
            pytest.skip("sandbox does not permit Unix-socket binding")
        finally:
            probe_path.unlink(missing_ok=True)
        context = multiprocessing.get_context("fork")
        process = context.Process(
            target=model_worker.serve_prestarted_model_worker,
            args=(socket_path, ready_file, busy_file),
        )
        process.start()
        try:
            deadline = time.monotonic() + 3.0
            while not ready_file.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready_file.is_file()
            assert process.pid is not None
            os.kill(process.pid, signal.SIGTERM)
            process.join(timeout=3.0)
            assert not process.is_alive()
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)

        assert process.exitcode == 0
        assert not ready_file.exists()
        assert not busy_file.exists()
        assert not socket_path.exists()


def test_dispatch_acknowledges_only_after_resident_worker_accepts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SuperExec should see acceptance only after authority reaches the worker."""
    connection = MagicMock()
    connection.__enter__.return_value = connection
    channel = MagicMock()
    connection.makefile.return_value.__enter__.return_value = channel
    monkeypatch.setattr(socket, "socket", Mock(return_value=connection))
    monkeypatch.setattr(
        model_worker_protocol,
        "read_message",
        Mock(
            side_effect=[
                {"event": "accepted"},
                {"event": "finished", "returncode": 0},
            ]
        ),
    )

    returncode = model_worker.dispatch_prestarted_model(
        ModelInvocation(
            token="task-token",
            runtime_api_address="runtime.example:9092",
            insecure=True,
            root_certificates_path=None,
        )
    )

    assert returncode == 0
    assert capsys.readouterr().out.strip() == FLWR_TASK_TOKEN_STDIN_ACKNOWLEDGEMENT
    connection.connect.assert_called_once()
    sent = channel.write.call_args.args[0]
    assert b"task-token" in sent


def test_dispatch_relays_typed_output_after_acceptance(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The exec-side process should reproduce resident stdout and stderr."""
    connection = MagicMock()
    connection.__enter__.return_value = connection
    channel = MagicMock()
    connection.makefile.return_value.__enter__.return_value = channel
    monkeypatch.setattr(socket, "socket", Mock(return_value=connection))
    monkeypatch.setattr(
        model_worker_protocol,
        "read_message",
        Mock(
            side_effect=[
                {"event": "accepted"},
                {"event": "output", "stream": "stdout", "data": "model out\n"},
                {"event": "output", "stream": "stderr", "data": "model err\n"},
                {"event": "finished", "returncode": 0},
            ]
        ),
    )

    returncode = model_worker.dispatch_prestarted_model(
        ModelInvocation("task-token", "runtime.example:9092", True, None)
    )

    captured = capsys.readouterr()
    assert returncode == 0
    assert captured.out == f"{FLWR_TASK_TOKEN_STDIN_ACKNOWLEDGEMENT}\nmodel out\n"
    assert captured.err == "model err\n"


def test_dispatch_rejects_output_before_acceptance(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only output owned by an accepted task may reach the exec streams."""
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.makefile.return_value.__enter__.return_value = MagicMock()
    monkeypatch.setattr(socket, "socket", Mock(return_value=connection))
    monkeypatch.setattr(
        model_worker_protocol,
        "read_message",
        Mock(return_value={"event": "output", "stream": "stdout", "data": "bad"}),
    )

    returncode = model_worker.dispatch_prestarted_model(
        ModelInvocation("task-token", "runtime.example:9092", True, None)
    )

    assert returncode == 1
    assert capsys.readouterr().out == ""


def test_prestarted_worker_reads_certificates_for_each_invocation(
    tmp_path: Path,
) -> None:
    """Certificate bytes should enter only the task-scoped Runtime client."""
    certificate_path = tmp_path / "runtime-ca.pem"
    certificate_path.write_bytes(b"test-ca")
    server, client = socket.socketpair()

    def run_once(
        _runtime_api_address: str,
        _token: str,
        _insecure: bool,
        _certificates: bytes | None,
        on_started: Callable[[], None],
    ) -> int:
        on_started()
        return 0

    run_once_mock = Mock(side_effect=run_once)
    result: list[int] = []
    thread = threading.Thread(
        target=lambda: result.append(
            model_worker._serve_connection(server, run_once_mock)
        )
    )
    thread.start()
    try:
        with client.makefile("rwb") as channel:
            model_worker_protocol.send_message(
                channel,
                {
                    "token": "task-token",
                    "runtime_api_address": "runtime.example:9092",
                    "insecure": False,
                    "root_certificates_path": str(certificate_path),
                },
            )
            assert model_worker_protocol.read_message(channel) == {"event": "accepted"}
            assert model_worker_protocol.read_message(channel) == {
                "event": "finished",
                "returncode": 0,
            }
    finally:
        client.close()
        server.close()
        thread.join(timeout=2.0)

    assert result == [0]
    call_args = run_once_mock.call_args.args
    assert call_args[:4] == (
        "runtime.example:9092",
        "task-token",
        False,
        b"test-ca",
    )
    assert callable(call_args[4])


def test_prestarted_worker_rejects_failure_before_acceptance() -> None:
    """Runtime setup failure should not emit an authority acknowledgement."""
    server, client = socket.socketpair()
    run_once = Mock(side_effect=RuntimeError("setup failed"))
    result: list[int] = []
    thread = threading.Thread(
        target=lambda: result.append(model_worker._serve_connection(server, run_once))
    )
    thread.start()
    try:
        with client.makefile("rwb") as channel:
            model_worker_protocol.send_message(
                channel,
                {
                    "token": "task-token",
                    "runtime_api_address": "runtime.example:9092",
                    "insecure": True,
                    "root_certificates_path": None,
                },
            )
            assert model_worker_protocol.read_message(channel) == {
                "event": "rejected",
                "reason": "Prestarted Model worker could not start task.",
            }
    finally:
        client.close()
        server.close()
        thread.join(timeout=2.0)

    assert result == [1]


def test_prestarted_worker_relays_redacted_bounded_output() -> None:
    """Resident output should be framed without leaking authority markers."""
    token = "task-token-that-must-not-be-relayed"
    server, client = socket.socketpair()

    def run_once(
        _runtime_api_address: str,
        invocation_token: str,
        _insecure: bool,
        _certificates: bytes | None,
        on_started: Callable[[], None],
    ) -> int:
        on_started()
        midpoint = len(invocation_token) // 2
        sys.stdout.write(f"before {invocation_token[:midpoint]}")
        sys.stdout.write(f"{invocation_token[midpoint:]} after\n")
        sys.stderr.write("model error\n")
        sys.stderr.write(FLWR_TASK_TOKEN_STDIN_ACKNOWLEDGEMENT)
        return 0

    result: list[int] = []
    thread = threading.Thread(
        target=lambda: result.append(model_worker._serve_connection(server, run_once))
    )
    thread.start()
    frames: list[dict[str, object]] = []
    try:
        with client.makefile("rwb") as channel:
            model_worker_protocol.send_message(
                channel,
                {
                    "token": token,
                    "runtime_api_address": "runtime.example:9092",
                    "insecure": True,
                    "root_certificates_path": None,
                },
            )
            while True:
                frame = model_worker_protocol.read_message(channel)
                frames.append(frame)
                if frame.get("event") == "finished":
                    break
    finally:
        client.close()
        server.close()
        thread.join(timeout=2.0)

    output = "".join(
        str(frame["data"]) for frame in frames if frame.get("event") == "output"
    )
    assert result == [0]
    assert frames[0] == {"event": "accepted"}
    assert frames[-1] == {"event": "finished", "returncode": 0}
    assert "before [REDACTED] after\n" in output
    assert "model error\n" in output
    assert token not in output
    assert FLWR_TASK_TOKEN_STDIN_ACKNOWLEDGEMENT not in output
    assert all(
        len(model_worker_protocol.encode_message(frame))
        <= model_worker_protocol.MAX_PROTOCOL_MESSAGE_BYTES
        for frame in frames
    )


def test_dispatcher_disconnect_does_not_abort_accepted_task() -> None:
    """Losing the exec reader must not change accepted-task execution."""
    server, client = socket.socketpair()
    started = threading.Event()
    release = threading.Event()

    def wait_for_release(
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

    run_once = Mock(side_effect=wait_for_release)
    result: list[int] = []
    thread = threading.Thread(
        target=lambda: result.append(model_worker._serve_connection(server, run_once))
    )
    thread.start()
    channel = client.makefile("rwb")
    model_worker_protocol.send_message(
        channel,
        {
            "token": "task-token",
            "runtime_api_address": "runtime.example:9092",
            "insecure": True,
            "root_certificates_path": None,
        },
    )
    assert model_worker_protocol.read_message(channel) == {"event": "accepted"}
    assert started.wait(timeout=1.0)
    channel.close()
    client.close()
    release.set()
    thread.join(timeout=2.0)
    server.close()

    assert result == [0]
    run_once.assert_called_once()


def test_saturated_output_channel_does_not_abort_task() -> None:
    """A blocked protocol writer should only drop logs, never block the task."""

    class BlockingChannel(BytesIO):
        """Block every write after the synchronous acceptance frame."""

        def __init__(self, initial_bytes: bytes) -> None:
            super().__init__(initial_bytes)
            self.writes = 0
            self.release = threading.Event()

        def write(self, data: Any, /) -> int:
            self.writes += 1
            if self.writes > 1:
                self.release.wait(timeout=2.0)
            return len(data)

        def flush(self) -> None:
            return

    invocation = (
        b'{"token":"task-token","runtime_api_address":"runtime:9092",'
        b'"insecure":true,"root_certificates_path":null}\n'
    )
    channel = BlockingChannel(invocation)

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
    result = model_worker._serve_channel(channel, Mock(side_effect=run_once))
    elapsed = time.monotonic() - started_at
    channel.release.set()

    assert result == 0
    assert elapsed < 1.5
