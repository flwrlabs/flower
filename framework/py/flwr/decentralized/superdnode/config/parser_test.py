"""Unit tests for SuperDNode run parser."""

from pathlib import Path

from flwr.decentralized.superdnode.config.parser import _parse_args_run


def test_parse_args_run_defaults() -> None:
    """Parser should expose expected deploy/simulation defaults."""
    parser = _parse_args_run()
    args = parser.parse_args(["--context", "ctx"])

    assert args.execution_mode == "simulation"
    assert args.timeout == 500
    assert args.nodeapps_pyproject == Path("pyproject.toml")


def test_parse_args_run_parses_simulation_flags() -> None:
    """Parser should include simulation-specific arguments."""
    parser = _parse_args_run()
    args = parser.parse_args(
        [
            "--context",
            "ctx",
            "--execution-mode",
            "simulation",
            "--nb-nodes",
            "12",
            "--sim-timeout",
            "44",
            "--no-enable-sampling",
        ]
    )

    assert args.nb_nodes == 12
    assert args.sim_timeout == 44
    assert args.enable_sampling is False
