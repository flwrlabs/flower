"""Convenience launcher for the PyTorch simulation example."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flwr.decentralized.superdnode.cli.flower_super_dnode import run as run_super_dnode


def main() -> None:
    """Run Flower Super DNode in simulation mode with local defaults."""
    parser = argparse.ArgumentParser(description="Run the simulation-pytorch example")
    parser.add_argument("--nb-nodes", type=int, default=20)
    parser.add_argument("--max-sim-time", type=float, default=250.0)
    parser.add_argument("--sim-timeout", type=int, default=120)
    parser.add_argument("--time-step-ms", type=int, default=100)
    parser.add_argument("--base-latency-ms", type=float, default=30.0)
    parser.add_argument("--jitter-factor", type=float, default=0.05)
    parser.add_argument("--verbose-sim", action="store_true")
    parser.add_argument("--multi-thread", action="store_true")
    args, passthrough_args = parser.parse_known_args(sys.argv[1:])

    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    sampling_config_path = Path(__file__).resolve().parents[1] / "config_sampling.json"

    run_super_dnode(
        [
            "--execution-mode",
            "simulation",
            "--nodeapps-pyproject",
            str(pyproject_path),
            "--sampling-config-file",
            str(sampling_config_path),
            "--nb-nodes",
            str(args.nb_nodes),
            "--max-sim-time",
            str(args.max_sim_time),
            "--sim-timeout",
            str(args.sim_timeout),
            "--time-step-ms",
            str(args.time_step_ms),
            "--base-latency-ms",
            str(args.base_latency_ms),
            "--jitter-factor",
            str(args.jitter_factor),
            *( ["--verbose-sim"] if args.verbose_sim else []),
            *( ["--multi-thread"] if args.multi_thread else []),
            *passthrough_args,
        ]
    )


if __name__ == "__main__":
    main()
