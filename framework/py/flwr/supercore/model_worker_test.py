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
import threading
import time
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from flwr.common.constant import FLWR_TASK_TOKEN_STDIN_ACKNOWLEDGEMENT

from . import model_worker
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

    assert model_worker._read_message(channel) == {"event": "accepted"}
    assert model_worker._read_message(channel) == {
        "event": "finished",
        "returncode": 0,
    }


def test_prestarted_worker_serves_one_invocation_and_cleans_up(
    tmp_path: Path,
) -> None:
    """The resident process should become busy, run once, and remove markers."""
    ready_file = tmp_path / "ready"
    busy_file = tmp_path / "busy"

    def run_once(
        runtime_api_address: str,
        token: str,
        insecure: bool,
        certificates: bytes | None,
    ) -> int:
        assert runtime_api_address == "runtime.example:9092"
        assert token == "task-token"
        assert insecure
        assert certificates is None
        assert busy_file.is_file()
        assert not ready_file.exists()
        return 0

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
        model_worker._send_message(  # pylint: disable=protected-access
            channel,
            {
                "token": "task-token",
                "runtime_api_address": "runtime.example:9092",
                "insecure": True,
                "root_certificates_path": None,
            },
        )
        assert model_worker._read_message(  # pylint: disable=protected-access
            channel
        ) == {"event": "accepted"}
        assert model_worker._read_message(  # pylint: disable=protected-access
            channel
        ) == {"event": "finished", "returncode": 0}
    thread.join(timeout=2.0)
    client_connection.close()

    assert result == [0]
    assert not thread.is_alive()
    run_model_once.assert_called_once_with(
        "runtime.example:9092", "task-token", True, None
    )
    assert not ready_file.exists()
    assert not busy_file.exists()


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
        model_worker,
        "_read_message",
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


def test_prestarted_worker_reads_certificates_for_each_invocation(
    tmp_path: Path,
) -> None:
    """Certificate bytes should enter only the task-scoped Runtime client."""
    certificate_path = tmp_path / "runtime-ca.pem"
    certificate_path.write_bytes(b"test-ca")
    server, client = socket.socketpair()
    run_once = Mock(return_value=0)
    result: list[int] = []
    thread = threading.Thread(
        target=lambda: result.append(model_worker._serve_connection(server, run_once))
    )
    thread.start()
    try:
        with client.makefile("rwb") as channel:
            model_worker._send_message(
                channel,
                {
                    "token": "task-token",
                    "runtime_api_address": "runtime.example:9092",
                    "insecure": False,
                    "root_certificates_path": str(certificate_path),
                },
            )
            assert model_worker._read_message(channel) == {"event": "accepted"}
            assert model_worker._read_message(channel) == {
                "event": "finished",
                "returncode": 0,
            }
    finally:
        client.close()
        server.close()
        thread.join(timeout=2.0)

    assert result == [0]
    run_once.assert_called_once_with(
        "runtime.example:9092", "task-token", False, b"test-ca"
    )
