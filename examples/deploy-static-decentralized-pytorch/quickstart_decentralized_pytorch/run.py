"""Scenario launchers for quickstart decentralized PyTorch modes."""

from __future__ import annotations

from flwr.decentralized.superdnode.cli.flower_super_dnode import run as run_super_dnode


def deploy_dynamic() -> None:
    run_super_dnode(
        [
            "--execution-mode",
            "deploy",
            "--config",
            "configs/deploy_dynamic.yaml",
            "--nodeapps-pyproject",
            "pyproject.toml",
            "--port",
            "9100",
            "--node-data-config-json",
            '{"partition-id": 0, "num-partitions": 1}',
        ]
    )


def deploy_static() -> None:
    run_super_dnode(
        [
            "--execution-mode",
            "deploy",
            "--config",
            "configs/deploy_static.yaml",
            "--nodeapps-pyproject",
            "pyproject.toml",
            "--node-name",
            "node_1",
            "--node-data-config-json",
            '{"partition-id": 0, "num-partitions": 4}',
        ]
    )


def simulation_dynamic_graph() -> None:
    run_super_dnode(
        [
            "--execution-mode",
            "simulation",
            "--sim-config",
            "configs/simulation_dynamic_graph.yaml",
            "--nodeapps-pyproject",
            "pyproject.toml",
        ]
    )


def simulation_static_graph_sampling() -> None:
    run_super_dnode(
        [
            "--execution-mode",
            "simulation",
            "--sim-config",
            "configs/simulation_static_graph_sampling.yaml",
            "--network-config-mode",
            "csr",
            "--enable-sampling",
            "--nodeapps-pyproject",
            "pyproject.toml",
        ]
    )
