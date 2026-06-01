# Copyright 2026 Inria (cyrille kenfack & davide frey). All Rights Reserved.
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
"""Unit tests for `flower_super_dnode` CLI module."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from flwr.decentralized.superdnode.cli.flower_super_dnode import (
    _load_nodeapps_from_pyproject,
    _parse_args_run,
    _run_deploy,
    _run_simulation,
    run,
)


def test_load_nodeapps_from_missing_pyproject_returns_empty(tmp_path: Path) -> None:
    """No pyproject -> no apps loaded."""
    missing = tmp_path / "pyproject.toml"
    assert _load_nodeapps_from_pyproject(missing) == []


def test_load_nodeapps_from_existing_pyproject_calls_factory(tmp_path: Path) -> None:
    """Existing pyproject delegates to nodeapp factory."""
    cfg = tmp_path / "pyproject.toml"
    cfg.write_text("[tool.flwr]\n")

    fake_app = MagicMock(name="app")
    with patch(
        "flwr.decentralized.superdnode.config.helper." "_create_nodeapps_from_pyproject",
        return_value={"a": fake_app},
    ) as factory:
        out = _load_nodeapps_from_pyproject(cfg)

    factory.assert_called_once_with(cfg)
    assert out == [fake_app]


def test_run_deploy_registers_loaded_apps_and_runs(tmp_path: Path) -> None:
    """Deploy mode registers autoloaded apps, runs node, and unregisters them."""
    app_a = SimpleNamespace(name="subject_a", subject="subject_a", node=None)
    app_b = SimpleNamespace(name="subject_b", subject="subject_b", node=None)

    args = SimpleNamespace(
        timeout=123,
        disable_nodeapps_autoload=False,
        nodeapps_pyproject=tmp_path / "pyproject.toml",
        node_data_config_json="",
    )

    runtime_node = MagicMock(name="runtime_node")
    runtime_node.to_dnode_kwargs.return_value = {
        "context": "ctx",
        "address": "0",
        "port": 1,
    }

    dnode = MagicMock(name="dnode")

    with (
        patch(
            "flwr.decentralized.superdnode.cli.flower_super_dnode._get_args_nodes",
            return_value=runtime_node,
        ),
        patch(
            "flwr.decentralized.superdnode.cli.flower_super_dnode._create_dnode",
            return_value=dnode,
        ),
        patch(
            "flwr.decentralized.superdnode.cli.flower_super_dnode."
            "_load_nodeapps_from_pyproject",
            return_value=[app_a, app_b],
        ),
    ):
        _run_deploy(args, ["--context", "ctx", "--port", "1"])

    dnode.create_node.assert_called_once()
    dnode.run.assert_called_once_with(timeout=123)
    dnode.register.assert_any_call(app_name="subject_a", app=app_a)
    dnode.register.assert_any_call(app_name="subject_b", app=app_b)
    dnode.unregister.assert_any_call(app_name="subject_a")
    dnode.unregister.assert_any_call(app_name="subject_b")


def test_run_deploy_skips_autoload_when_disabled(tmp_path: Path) -> None:
    """Disable flag prevents loading NodeApps from pyproject."""
    args = SimpleNamespace(
        timeout=5,
        disable_nodeapps_autoload=True,
        nodeapps_pyproject=tmp_path / "pyproject.toml",
        node_data_config_json="",
    )

    runtime_node = MagicMock(name="runtime_node")
    runtime_node.to_dnode_kwargs.return_value = {
        "context": "ctx",
        "address": "0",
        "port": 1,
    }

    dnode = MagicMock(name="dnode")

    with (
        patch(
            "flwr.decentralized.superdnode.cli.flower_super_dnode._get_args_nodes",
            return_value=runtime_node,
        ),
        patch(
            "flwr.decentralized.superdnode.cli.flower_super_dnode._create_dnode",
            return_value=dnode,
        ),
        patch(
            "flwr.decentralized.superdnode.cli.flower_super_dnode."
            "_load_nodeapps_from_pyproject",
        ) as loader,
    ):
        _run_deploy(args, ["--context", "ctx", "--port", "1"])

    loader.assert_not_called()
    dnode.run.assert_called_once_with(timeout=5)


def test_parse_args_run_includes_execution_mode_option() -> None:
    """CLI parser supports deploy/simulation execution mode values."""
    parser = _parse_args_run()
    args = parser.parse_args(["--execution-mode", "simulation"])
    assert args.execution_mode == "simulation"


def test_run_simulation_dispatches_to_run_simulation() -> None:
    """Simulation mode should dispatch to _run_simulation."""
    with patch(
        "flwr.decentralized.superdnode.cli.flower_super_dnode._run_simulation"
    ) as run_simulation:
        run(["--execution-mode", "simulation"])

    run_simulation.assert_called_once()


def test_run_simulation_respects_network_config_mode_override(tmp_path: Path) -> None:
    """Use explicit `network_config_mode` when provided by args."""
    args = SimpleNamespace(
        disable_nodeapps_autoload=True,
        nodeapps_pyproject=tmp_path / "pyproject.toml",
        base_latency_ms=30.0,
        jitter_factor=0.1,
        failure_probability=0.0,
        recovery_time=10.0,
        sync_node_count=0,
        sync_interval_ms=500,
        max_drift_ms=0,
        time_step_ms=100,
        max_sim_time=5.0,
        real_time_factor=1.0,
        verbose_sim=False,
        network_config_mode="csr",
        enable_sampling=True,
        sampling_config_file="config_sampling.json",
        nb_nodes=4,
        sampling_algorithm="gbps",
        topology_kind="ring",
        topology_seed=42,
        random_mode="exact",
        random_send_to=1,
        random_receive_from=1,
        random_min_send_to=None,
        random_max_send_to=2,
        random_min_receive_from=None,
        random_max_receive_from=2,
        sampling_view_size=4,
        sampling_heal=0,
        sampling_swap=0,
        sampling_selection_policy="old",
        sampling_propagation_policy="pushpull",
        sampling_delay=2,
        sampling_age=1,
        sampling_sampler_size=8,
        sampling_alpha=0.5,
        sampling_beta=0.5,
        sampling_refresh=1,
        sim_timeout=10,
        multi_thread=False,
        sampling_period=1000,
    )

    with (
        patch(
            "flwr.decentralized.superdnode.cli.flower_super_dnode._build_sim_config",
            return_value=MagicMock(name="sim_config"),
        ),
        patch(
            "flwr.decentralized.superdnode.cli.flower_super_dnode._build_sampling_config",
            return_value=MagicMock(name="sampling_config"),
        ) as build_sampling,
        patch(
            "flwr.decentralized.superdnode.cli.flower_super_dnode._run_nodeapp_simulation"
        ),
    ):
        _run_simulation(args)

    kwargs = build_sampling.call_args.kwargs
    assert kwargs["network_config_mode"] == "csr"
    assert kwargs["attach_sampling_to_csr"] is True


def test_run_reports_missing_decentralized_extra() -> None:
    """Raise a friendly install hint when nodemanager is missing."""
    with patch(
        "flwr.decentralized.superdnode.cli.flower_super_dnode._run_simulation",
        side_effect=ModuleNotFoundError("No module named 'nodemanager'"),
    ):
        try:
            run(["--execution-mode", "simulation"])
            raised = False
        except SystemExit as exc:
            raised = True
            assert "flwr[decentralized]" in str(exc)

    assert raised
