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
"""Tests for the prestarted Connector worker adapter."""

# pylint: disable=protected-access

import importlib
import subprocess
import sys
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest

from flwr.supercore.typing import JSONObject

from . import connector_worker, task_worker, task_worker_protocol
from .connector_worker import ConnectorInvocation

run_connector_module = importlib.import_module(
    "flwr.supercore.task_process.connector.run_connector"
)


def _payload() -> JSONObject:
    return {
        "token": "task-token",
        "runtime_api_address": "runtime.example:9092",
        "insecure": True,
        "root_certificates_path": None,
    }


def test_connector_worker_serve_and_dispatch_interfaces(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Connector serve and dispatch should delegate to the shared worker."""
    assert ConnectorInvocation.from_payload(_payload()) == ConnectorInvocation(
        "task-token", "runtime.example:9092", True, None
    )
    socket_path = tmp_path / "connector.sock"
    ready_file = tmp_path / "ready"
    busy_file = tmp_path / "busy"
    serve = Mock(return_value=0)
    dispatch = Mock(return_value=0)
    create_client = Mock()
    monkeypatch.setattr(task_worker, "serve_prestarted_worker", serve)
    monkeypatch.setattr(task_worker, "dispatch_prestarted_task", dispatch)
    monkeypatch.setattr(run_connector_module, "_create_runtime_client", create_client)

    assert (
        connector_worker.serve_prestarted_connector_worker(
            socket_path, ready_file, busy_file
        )
        == 0
    )
    run_once = serve.call_args.args[3]
    assert run_once is run_connector_module.run_connector_once
    assert serve.call_args.args[4:] == (
        ConnectorInvocation.from_payload,
        "Connector",
    )
    create_client.assert_not_called()

    invocation = ConnectorInvocation("task-token", "runtime.example:9092", True, None)
    assert connector_worker.dispatch_prestarted_connector(invocation, socket_path) == 0
    dispatch.assert_called_once_with(_payload(), socket_path, "Connector")


def test_connector_worker_relays_captured_subprocess_output() -> None:
    """Pipe-captured subprocess output should retain task attribution."""
    channel = BytesIO(task_worker_protocol.encode_message(_payload()))

    def run_once(
        _runtime_api_address: str,
        _token: str,
        _insecure: bool,
        _certificates: bytes | None,
        on_started: Callable[[], None],
    ) -> int:
        on_started()
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", "print('browser child output')"],
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout, end="")
        return 0

    assert (
        task_worker._serve_channel(
            channel,
            run_once,
            ConnectorInvocation.from_payload,
            "Connector",
        )
        == 0
    )

    responses = BytesIO(channel.getvalue().partition(b"\n")[2])
    frames = []
    while True:
        frame = task_worker_protocol.read_message(responses)
        frames.append(frame)
        if frame.get("event") == "finished":
            break
    assert frames[0] == {"event": "accepted"}
    assert "browser child output\n" in "".join(
        str(frame.get("data", "")) for frame in frames
    )
    assert frames[-1] == {"event": "finished", "returncode": 0}


@pytest.mark.parametrize(
    "payload",
    [{"unexpected": True}, {**_payload(), "extra": "field"}],
)
def test_connector_invocation_rejects_unexpected_fields(
    payload: dict[str, object],
) -> None:
    """Connector payload validation should reject incomplete or extra fields."""
    with pytest.raises(ValueError, match="unexpected fields"):
        ConnectorInvocation.from_payload(payload)
