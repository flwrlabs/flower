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
"""Tests for the FAB-specific prestarted AgentApp worker."""

import importlib
from pathlib import Path
from unittest.mock import Mock

import pytest

from . import agentapp_worker, task_worker
from .agentapp_worker import AgentAppInvocation

run_agentapp_module = importlib.import_module(
    "flwr.supercore.task_process.agent.run_agentapp"
)

FAB_HASH = "a" * 64


def _payload(fab_hash: str = FAB_HASH) -> dict[str, object]:
    return {
        "token": "task-token",
        "runtime_api_address": "runtime.example:9092",
        "insecure": True,
        "root_certificates_path": None,
        "fab_hash": fab_hash,
    }


def test_agentapp_worker_preloads_before_serving_exact_fab(  # pylint: disable=too-many-locals
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Publish no worker until preload succeeds and require its exact FAB."""
    fab_path = tmp_path / "mounted-app"
    socket_path = tmp_path / "agentapp.sock"
    ready_file = tmp_path / "ready"
    busy_file = tmp_path / "busy"
    preloaded = Mock(fab_hash=FAB_HASH)
    calls: list[str] = []

    def preload_agentapp(*_: object) -> Mock:
        calls.append("preload")
        return preloaded

    def serve_worker(*_: object) -> int:
        calls.append("serve")
        return 0

    preload = Mock(side_effect=preload_agentapp)
    run_agentapp_once = Mock(return_value=0)
    serve = Mock(side_effect=serve_worker)
    monkeypatch.setattr(run_agentapp_module, "preload_agentapp", preload)
    monkeypatch.setattr(run_agentapp_module, "run_agentapp_once", run_agentapp_once)
    monkeypatch.setattr(task_worker, "serve_prestarted_worker", serve)

    assert (
        agentapp_worker.serve_prestarted_agentapp_worker(
            FAB_HASH, fab_path, socket_path, ready_file, busy_file
        )
        == 0
    )
    assert calls == ["preload", "serve"]
    preload.assert_called_once_with(fab_path, FAB_HASH)
    run_once = serve.call_args.args[3]
    parse_invocation = serve.call_args.args[4]
    secure_payload = {
        **_payload(),
        "insecure": False,
        "root_certificates_path": "/runtime-ca.pem",
    }
    assert parse_invocation(secure_payload).fab_hash == FAB_HASH
    with pytest.raises(ValueError, match="does not match"):
        parse_invocation(_payload("b" * 64))

    on_started = Mock()
    assert run_once("runtime.example:9092", "token", False, b"ca", on_started) == 0
    run_agentapp_once.assert_called_once_with(
        "runtime.example:9092",
        "token",
        False,
        b"ca",
        on_started,
        preloaded=preloaded,
        certificates_path="/runtime-ca.pem",
    )


def test_agentapp_dispatch_includes_fab_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The lightweight dispatcher should send the selected task FAB identity."""
    dispatch = Mock(return_value=0)
    monkeypatch.setattr(task_worker, "dispatch_prestarted_task", dispatch)
    socket_path = tmp_path / "agentapp.sock"
    invocation = AgentAppInvocation(
        "task-token", "runtime.example:9092", True, None, FAB_HASH
    )

    assert agentapp_worker.dispatch_prestarted_agentapp(invocation, socket_path) == 0
    dispatch.assert_called_once_with(_payload(), socket_path, "AgentApp")


def test_agentapp_dispatch_rejects_runtime_dependency_installation() -> None:
    """Prestarted apps require deployment-provisioned dependencies."""
    with pytest.raises(SystemExit):
        agentapp_worker._parse_args().parse_args(  # pylint: disable=protected-access
            [
                "dispatch",
                "--runtime-api-address",
                "runtime.example:9092",
                "--fab-hash",
                FAB_HASH,
                "--token",
                "task-token",
                "--allow-runtime-dependency-installation",
            ]
        )


@pytest.mark.parametrize("fab_hash", ["short", "A" * 64])
def test_agentapp_invocation_requires_full_sha256(fab_hash: str) -> None:
    """Reject malformed or noncanonical FAB routing identities."""
    with pytest.raises(ValueError, match="full SHA-256"):
        AgentAppInvocation.from_payload(_payload(fab_hash))
