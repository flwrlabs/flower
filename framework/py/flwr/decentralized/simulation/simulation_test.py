"""Unit tests for decentralized simulation helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from flwr.decentralized.simulation.simulation import (
    build_sampling_config,
    build_sim_config,
    run_nodeapp_simulation,
)


def test_build_sim_config_sets_core_values() -> None:
    """Create nested simulation config with provided scalar values."""
    cfg = build_sim_config(max_sim_time_seconds=12.5, time_step_ms=50)

    assert cfg.max_sim_time_seconds == 12.5
    assert cfg.time_step_ms == 50


def test_build_sampling_config_rejects_unknown_algorithm() -> None:
    """Reject unsupported sampling algorithm names."""
    with pytest.raises(ValueError, match="Unsupported sampling_algorithm"):
        build_sampling_config(sampling_algorithm="unknown")


def test_build_sampling_config_csr_requires_nb_nodes() -> None:
    """CSR network mode requires explicit node count."""
    with pytest.raises(ValueError, match="nb_nodes is required"):
        build_sampling_config(network_config_mode="csr", nb_nodes=None)


def test_build_sampling_config_csr_random_range_calls_graph_builder() -> None:
    """Build CSR config through simulation graph helper in random/range mode."""
    with patch(
        "flwr.decentralized.simulation.simulation.generate_simulation_csr",
        return_value=("csr-matrix", {}),
    ) as generate_csr:
        cfg = build_sampling_config(
            network_config_mode="csr",
            nb_nodes=5,
            topology_kind="random",
            random_mode="range",
            random_max_send_to=3,
            random_max_receive_from=4,
        )

    generate_csr.assert_called_once()
    assert cfg.config_file == "config_sampling.json"


def test_run_nodeapp_simulation_builds_instances_and_calls_engine() -> None:
    """Clone app instances and invoke nodemanager simulation once."""
    app = MagicMock(name="app")
    app.subject = "subject-a"
    app.for_node.side_effect = lambda i, n: f"instance-{i}-of-{n}"

    config = SimpleNamespace(max_sim_time_seconds=5.0)
    sampling_cfg = object()

    with patch(
        "flwr.decentralized.simulation.simulation.run_discrete_event_simulation"
    ) as run_sim:
        run_nodeapp_simulation(
            nb_nodes=3,
            apps=[app],
            config=config,
            timeout=8,
            multi_thread=True,
            sampling_period=123,
            sampling_config=sampling_cfg,
            enable_sampling=True,
        )

    assert app.for_node.call_count == 3
    run_sim.assert_called_once()
    kwargs = run_sim.call_args.kwargs
    assert kwargs["applications"]["subject-a"] == [
        "instance-0-of-3",
        "instance-1-of-3",
        "instance-2-of-3",
    ]
    assert kwargs["sampling_config"] is sampling_cfg
    assert kwargs["sampling_period"] == 123
    assert kwargs["enable_sampling"] is True
    assert kwargs["multi_thread"] is True
