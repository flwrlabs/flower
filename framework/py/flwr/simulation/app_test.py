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
"""Tests for simulation runtime wiring."""


import importlib
import unittest
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import pytest

from flwr.common import Context, RecordDict
from flwr.common.constant import SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS, SubStatus
from flwr.common.serde import context_to_proto, fab_to_proto, run_to_proto
from flwr.common.typing import Fab, Run
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    PullTaskInputResponse,
    PushTaskOutputRequest,
)
from flwr.proto.federation_config_pb2 import SimulationConfig  # pylint: disable=E0611
from flwr.server.superlink.fleet.vce.metrics import VceMetrics

from .app import _parse_args_run_flwr_simulation, run_simulation_process

simulation_app_module = importlib.import_module("flwr.simulation.app")
_TEST_CLIENTAPP_RUNTIME = 7.89


class TestRunSimulationProcess(unittest.TestCase):
    """Tests for `run_simulation_process`."""

    @patch("flwr.simulation.app.flwr_exit")
    @patch("flwr.simulation.app.register_signal_handlers")
    @patch("flwr.simulation.app.SimulationIoConnection")
    def test_run_simulation_process_passes_token_to_connection(
        self,
        mock_connection_cls: Mock,
        _mock_register_signal_handlers: Mock,
        mock_flwr_exit: Mock,
    ) -> None:
        """`run_simulation_process` should pass token into SimulationIoConnection."""
        mock_conn = Mock()
        mock_conn.configure_mock(
            **{"_stub.PullTaskInput.side_effect": RuntimeError("boom")}
        )
        mock_connection_cls.return_value = mock_conn

        run_simulation_process(
            serverappio_api_address="127.0.0.1:9091",
            log_queue=Queue(),
            insecure=True,
            token="test-token",
        )

        mock_connection_cls.assert_called_once_with(
            serverappio_api_address="127.0.0.1:9091",
            insecure=True,
            root_certificates=None,
            token="test-token",
        )
        mock_flwr_exit.assert_called_once()


def _test_context() -> Context:
    """Return a minimal Simulation Runtime context."""
    return Context(
        run_id=1234,
        node_id=0,
        node_config={},
        state=RecordDict(),
        run_config={},
    )


def _add_test_metrics(metrics: VceMetrics) -> None:
    """Add deterministic metrics to a VCE metrics accumulator."""
    metrics.add_clientapp_runtime(_TEST_CLIENTAPP_RUNTIME)


def _patch_run_simulation_process_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, context: Context
) -> list[PushTaskOutputRequest]:
    """Patch external process dependencies and capture pushed task output."""
    run = Run.create_empty(run_id=1234)
    run.fab_hash = "fab-hash"
    fab = Fab(hash_str="fab-hash", content=b"fab-content", verifications={})
    pull_response = PullTaskInputResponse(
        context=context_to_proto(context),
        run=run_to_proto(run),
        fab=fab_to_proto(fab),
        federation_config=SimulationConfig(num_supernodes=1),
    )
    pushed_requests: list[PushTaskOutputRequest] = []
    mock_conn = Mock()
    # pylint: disable=protected-access,unnecessary-lambda
    mock_conn._stub.PullTaskInput.return_value = pull_response
    mock_conn._stub.PushTaskOutput.side_effect = lambda req: pushed_requests.append(req)
    mock_conn._retry_invoker = SimpleNamespace(max_tries=3)
    # pylint: enable=protected-access,unnecessary-lambda
    heartbeat_sender = Mock()
    heartbeat_sender.is_running = True

    project_config = {
        "tool": {
            "flwr": {
                "app": {
                    "components": {
                        "clientapp": "client:app",
                        "serverapp": "server:app",
                    }
                }
            }
        }
    }
    patches = {
        "SimulationIoConnection": Mock(return_value=mock_conn),
        "register_signal_handlers": Mock(),
        "HeartbeatSender": Mock(return_value=heartbeat_sender),
        "make_task_heartbeat_fn_grpc": lambda _stub: lambda: None,
        "start_log_uploader": lambda **_kwargs: None,
        "install_from_fab": lambda *_args, **_kwargs: None,
        "get_fab_metadata": lambda _content: ("app", "1.0.0"),
        "get_project_dir": lambda *_args: tmp_path,
        "get_project_config": lambda _path: project_config,
        "get_fused_config_from_dir": lambda *_args: {},
        "cleanup_app_runtime_environment": lambda _path: None,
        "event": lambda *_args, **_kwargs: None,
        "flwr_exit": Mock(),
    }
    for name, value in patches.items():
        monkeypatch.setattr(simulation_app_module, name, value)

    return pushed_requests


@pytest.mark.parametrize("raise_after_metrics", [False, True])
def test_run_simulation_process_pushes_simulation_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raise_after_metrics: bool
) -> None:
    """Simulation process should report collected run metrics to SuperLink."""
    context = _test_context()
    pushed_requests = _patch_run_simulation_process_dependencies(
        monkeypatch, tmp_path, context
    )

    def _run_simulation(**kwargs: object) -> SimpleNamespace:
        metrics = cast(VceMetrics, kwargs["metrics"])
        _add_test_metrics(metrics)
        if raise_after_metrics:
            raise RuntimeError("simulation failed after processing messages")
        return SimpleNamespace(context=context, metrics=metrics)

    monkeypatch.setattr(
        simulation_app_module,
        "_run_simulation",
        _run_simulation,
    )

    run_simulation_process(
        serverappio_api_address="127.0.0.1:9091",
        log_queue=Queue(),
        insecure=True,
        token="test-token",
        runtime_dependency_install=False,
    )

    assert len(pushed_requests) == 1
    out_req = pushed_requests[0]
    expected_sub_status = (
        SubStatus.FAILED if raise_after_metrics else SubStatus.COMPLETED
    )
    assert out_req.sub_status == expected_sub_status
    if raise_after_metrics:
        assert "simulation failed after processing messages" in out_req.details
    assert out_req.clientapp_runtime == _TEST_CLIENTAPP_RUNTIME


def test_parse_flwr_simulation_requires_token() -> None:
    """The simulation process CLI should require a token."""
    with pytest.raises(SystemExit):
        _parse_args_run_flwr_simulation().parse_args([])


def test_parse_flwr_simulation_rejects_run_once() -> None:
    """The removed deprecated flag should no longer parse."""
    with pytest.raises(SystemExit):
        _parse_args_run_flwr_simulation().parse_args(
            ["--token", "test-token", "--run-once"]
        )


def test_parse_flwr_simulation_parses_tokenized_invocation() -> None:
    """The simulation process CLI should still parse the supported flags."""
    args = _parse_args_run_flwr_simulation().parse_args(
        [
            "--token",
            "test-token",
            "--insecure",
            "--parent-pid",
            "1234",
            "--allow-runtime-dependency-installation",
        ]
    )

    assert args.serverappio_api_address == SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS
    assert args.token == "test-token"
    assert args.insecure is True
    assert args.parent_pid == 1234
    assert args.runtime_dependency_install is True


def test_flwr_simulation_parses_args_before_mirroring_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Argument parsing should happen before stdout/stderr redirection."""

    class _Parser:
        def parse_args(self) -> SimpleNamespace:
            """Raise a parser error before any side effects happen."""
            raise SystemExit(2)

    calls: list[str] = []

    monkeypatch.setattr(
        simulation_app_module, "_parse_args_run_flwr_simulation", _Parser
    )
    monkeypatch.setattr(
        simulation_app_module,
        "mirror_output_to_queue",
        lambda *_args, **_kwargs: calls.append("mirror"),
    )

    with pytest.raises(SystemExit):
        simulation_app_module.flwr_simulation()

    assert not calls


def test_flwr_simulation_forwards_cli_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The simulation CLI should forward parsed args to the runtime."""
    args = SimpleNamespace(
        insecure=True,
        serverappio_api_address="127.0.0.1:9091",
        token="test-token",
        root_certificates=None,
        parent_pid=321,
        runtime_dependency_install=True,
    )
    calls: list[str] = []
    captured: dict[str, object] = {}

    class _Parser:
        def parse_args(self) -> SimpleNamespace:
            """Return a fixed namespace for CLI forwarding tests."""
            return args

    def _mirror_output_to_queue(*_args: object, **_kwargs: object) -> None:
        calls.append("mirror")

    def _restore_output() -> None:
        calls.append("restore")

    def _run_simulation_process(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        simulation_app_module, "_parse_args_run_flwr_simulation", _Parser
    )
    monkeypatch.setattr(
        simulation_app_module, "mirror_output_to_queue", _mirror_output_to_queue
    )
    monkeypatch.setattr(simulation_app_module, "restore_output", _restore_output)
    monkeypatch.setattr(
        simulation_app_module, "run_simulation_process", _run_simulation_process
    )

    simulation_app_module.flwr_simulation()

    assert calls == ["mirror", "restore"]
    assert captured["serverappio_api_address"] == "127.0.0.1:9091"
    assert captured["token"] == "test-token"
    assert captured["certificates"] is None
    assert captured["parent_pid"] == 321
    assert captured["runtime_dependency_install"] is True


def test_flwr_simulation_forwards_token_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The simulation CLI should read and forward token file contents."""
    token_file = tmp_path / "token"
    token_file.write_text("test-token\n", encoding="utf-8")
    args = SimpleNamespace(
        insecure=True,
        serverappio_api_address="127.0.0.1:9091",
        token=None,
        token_file=str(token_file),
        root_certificates=None,
        parent_pid=321,
        runtime_dependency_install=True,
    )
    captured: dict[str, object] = {}

    class _Parser:
        def parse_args(self) -> SimpleNamespace:
            """Return a fixed namespace for CLI forwarding tests."""
            return args

    monkeypatch.setattr(
        simulation_app_module, "_parse_args_run_flwr_simulation", _Parser
    )
    monkeypatch.setattr(
        simulation_app_module,
        "mirror_output_to_queue",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(simulation_app_module, "restore_output", lambda: None)
    monkeypatch.setattr(
        simulation_app_module,
        "run_simulation_process",
        lambda **kwargs: captured.update(kwargs),
    )

    simulation_app_module.flwr_simulation()

    assert captured["token"] == "test-token"
