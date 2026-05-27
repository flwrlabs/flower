"""Unit tests for SuperDNode config helper functions."""

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from flwr.decentralized.superdnode.config.helper import (
    _apply_simulation_config_overrides,
    _load_nodeapps_from_pyproject,
    _load_simulation_config_file,
    _strip_superdnode_only_args,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-config", default=None)
    parser.add_argument("--nb-nodes", type=int, default=10)
    parser.add_argument("--sim-timeout", type=int, default=300)
    return parser


def test_load_simulation_config_file_unsupported_extension_raises(tmp_path: Path) -> None:
    """Reject unknown simulation config file extensions."""
    config = tmp_path / "sim.json"
    config.write_text("{}")

    with pytest.raises(ValueError, match="Unsupported simulation config format"):
        _load_simulation_config_file(config)


def test_apply_simulation_config_overrides_respects_cli_precedence(tmp_path: Path) -> None:
    """CLI flags should override values loaded from sim config file."""
    sim_cfg = tmp_path / "sim.toml"
    sim_cfg.write_text("[simulation]\nnb_nodes = 42\ntimeout = 123\n")

    parser = _build_parser()
    args = parser.parse_args(["--sim-config", str(sim_cfg), "--nb-nodes", "7"])

    updated = _apply_simulation_config_overrides(
        parser,
        args,
        ["--sim-config", str(sim_cfg), "--nb-nodes", "7"],
    )

    assert updated.nb_nodes == 7
    assert updated.sim_timeout == 123


def test_strip_superdnode_only_args_keeps_node_args() -> None:
    """Keep deploy-node args while stripping SuperDNode-only options."""
    argv = [
        "--execution-mode",
        "simulation",
        "--nb-nodes",
        "5",
        "--timeout",
        "10",
        "--context",
        "ctx",
        "--address",
        "0.0.0.0",
        "--port",
        "9001",
    ]

    stripped = _strip_superdnode_only_args(argv)

    assert stripped == ["--context", "ctx", "--address", "0.0.0.0", "--port", "9001"]


def test_load_nodeapps_from_pyproject_missing_file_returns_empty(tmp_path: Path) -> None:
    """Return empty list when pyproject file does not exist."""
    missing = tmp_path / "pyproject.toml"

    assert _load_nodeapps_from_pyproject(missing) == []


def test_load_nodeapps_from_pyproject_existing_file_loads_values(tmp_path: Path) -> None:
    """Load NodeApp values from factory mapping when file exists."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[tool.flwr]\n")

    app_a = SimpleNamespace(name="a")
    app_b = SimpleNamespace(name="b")

    with patch(
        "flwr.decentralized.superdnode.config.helper.create_nodeapps_from_pyproject",
        return_value={"a": app_a, "b": app_b},
    ):
        loaded = _load_nodeapps_from_pyproject(pyproject)

    assert loaded == [app_a, app_b]
