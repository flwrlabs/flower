"""Unit tests for simulation CLI arg registration."""

import argparse

from flwr.decentralized.simulation.args import add_simulation_args


def test_add_simulation_args_registers_defaults() -> None:
    """Simulation parser should expose stable defaults."""
    parser = argparse.ArgumentParser()
    add_simulation_args(parser)
    args = parser.parse_args([])

    assert args.nb_nodes == 10
    assert args.sim_timeout == 300
    assert args.enable_sampling is True
    assert args.topology_kind == "ring"


def test_add_simulation_args_boolean_optional_flag_parsing() -> None:
    """BooleanOptionalAction should support explicit disable flag."""
    parser = argparse.ArgumentParser()
    add_simulation_args(parser)
    args = parser.parse_args(["--no-enable-sampling"])

    assert args.enable_sampling is False
