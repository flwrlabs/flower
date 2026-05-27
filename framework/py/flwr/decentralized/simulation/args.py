"""CLI argument helpers for Super DNode simulation mode."""

import argparse


def add_simulation_args(parser: argparse.ArgumentParser) -> None:
	"""Register simulation-specific arguments on *parser*.

	All arguments are grouped under "simulation" for help readability.
	They are used by :func:`flwr.decentralized.superdnode.cli.flower_super_dnode._run_simulation`
	to build a :class:`~flwr.decentralized.simulation.config.DiscreteEventSimConfig`.
	"""
	grp = parser.add_argument_group("simulation")

	# ── Core ──────────────────────────────────────────────────────────────
	grp.add_argument(
		"--sim-config",
		type=str,
		default=None,
		help=(
			"Path to a YAML (.yaml/.yml) or TOML (.toml) simulation configuration file. "
			"CLI flags override values from this file."
		),
	)
	grp.add_argument(
		"--nb-nodes",
		type=int,
		default=10,
		help="Number of virtual nodes to instantiate in the simulation.",
	)
	grp.add_argument(
		"--sim-timeout",
		type=int,
		default=300,
		help="Wall-clock timeout before the simulation is forcibly stopped.",
	)
	grp.add_argument(
		"--max-sim-time",
		type=float,
		default=360000.0,
		help="Maximum simulated time in seconds.",
	)
	grp.add_argument(
		"--time-step-ms",
		type=int,
		default=100,
		help="Simulated time step in milliseconds.",
	)
	grp.add_argument(
		"--real-time-factor",
		type=float,
		default=1.0,
		help=(
			"Ratio of simulated time to wall-clock time.  "
			"Values > 1 run faster than real time; < 1 run slower."
		),
	)
	grp.add_argument(
		"--multi-thread",
		action="store_true",
		default=False,
		help="Run virtual nodes in separate threads (default: sequential).",
	)
	grp.add_argument(
		"--verbose-sim",
		action="store_true",
		default=False,
		help="Enable verbose logging from the simulation engine.",
	)
	grp.add_argument(
		"--sampling-period",
		type=int,
		default=1000,
		help="Peer-sampling period in milliseconds.",
	)
	grp.add_argument(
		"--enable-sampling",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Enable peer-sampling during the simulation.",
	)
	grp.add_argument(
		"--network-config-mode",
		type=str,
		choices=["sampling", "csr"],
		default=None,
		help=(
			"Deprecated: network mode is now inferred from --enable-sampling. "
			"Kept for backward compatibility."
		),
	)
	grp.add_argument(
		"--sampling-config-file",
		type=str,
		default="config_sampling.json",
		help="Path to the generated sampling/network JSON configuration file.",
	)
	grp.add_argument(
		"--sampling-algorithm",
		type=str,
		choices=["gbps", "brahams", "basalt"],
		default="gbps",
		help="Sampling algorithm used when --enable-sampling is true.",
	)
	grp.add_argument(
		"--sampling-view-size",
		type=int,
		default=4,
		help="View size used by the sampling algorithm.",
	)
	grp.add_argument(
		"--sampling-heal",
		type=int,
		default=0,
		help="GBPS heal parameter.",
	)
	grp.add_argument(
		"--sampling-swap",
		type=int,
		default=0,
		help="GBPS swap parameter.",
	)
	grp.add_argument(
		"--sampling-selection-policy",
		type=str,
		choices=["old", "rand"],
		default="old",
		help="GBPS selection policy.",
	)
	grp.add_argument(
		"--sampling-propagation-policy",
		type=str,
		choices=["push", "pushpull"],
		default="pushpull",
		help="GBPS propagation policy.",
	)
	grp.add_argument(
		"--sampling-delay",
		type=int,
		default=2,
		help="Sampling delay parameter.",
	)
	grp.add_argument(
		"--sampling-age",
		type=int,
		default=1,
		help="GBPS age parameter.",
	)
	grp.add_argument(
		"--sampling-sampler-size",
		type=int,
		default=8,
		help="Brahams sampler_size parameter.",
	)
	grp.add_argument(
		"--sampling-alpha",
		type=float,
		default=0.5,
		help="Brahams alpha parameter.",
	)
	grp.add_argument(
		"--sampling-beta",
		type=float,
		default=0.5,
		help="Brahams beta parameter.",
	)
	grp.add_argument(
		"--sampling-refresh",
		type=int,
		default=1,
		help="Basalt refresh parameter.",
	)

	# ── CSR topology generation ───────────────────────────────────────────
	grp.add_argument(
		"--topology-kind",
		type=str,
		choices=["ring", "star", "fullconnected", "random"],
		default="ring",
		help="Topology kind used when --network-config-mode=csr.",
	)
	grp.add_argument(
		"--topology-seed",
		type=int,
		default=42,
		help="Random seed for CSR topology generation.",
	)
	grp.add_argument(
		"--random-mode",
		type=str,
		choices=["exact", "range"],
		default="exact",
		help="Random topology mode for --topology-kind=random.",
	)
	grp.add_argument(
		"--random-send-to",
		type=int,
		default=1,
		help="Exact random mode: number of outgoing edges.",
	)
	grp.add_argument(
		"--random-receive-from",
		type=int,
		default=1,
		help="Exact random mode: number of incoming edges.",
	)
	grp.add_argument(
		"--random-min-send-to",
		type=int,
		default=None,
		help="Range random mode: min outgoing edges (optional).",
	)
	grp.add_argument(
		"--random-max-send-to",
		type=int,
		default=2,
		help="Range random mode: max outgoing edges.",
	)
	grp.add_argument(
		"--random-min-receive-from",
		type=int,
		default=None,
		help="Range random mode: min incoming edges (optional).",
	)
	grp.add_argument(
		"--random-max-receive-from",
		type=int,
		default=2,
		help="Range random mode: max incoming edges.",
	)

	# ── Latency ───────────────────────────────────────────────────────────
	grp.add_argument(
		"--base-latency-ms",
		type=float,
		default=50.0,
		help="Base message latency in milliseconds.",
	)
	grp.add_argument(
		"--jitter-factor",
		type=float,
		default=0.1,
		help=(
			"Relative jitter applied to base latency "
			"(e.g. 0.1 = ±10 %% of base latency)."
		),
	)

	# ── Disconnections ────────────────────────────────────────────────────
	grp.add_argument(
		"--failure-probability",
		type=float,
		default=0.0,
		help="Probability [0, 1] that a node fails at each time step.",
	)
	grp.add_argument(
		"--recovery-time",
		type=float,
		default=10.0,
		help="Expected recovery time (seconds) after a node disconnection.",
	)

	# ── Synchronisation ───────────────────────────────────────────────────
	grp.add_argument(
		"--sync-node-count",
		type=int,
		default=0,
		help=(
			"Number of nodes that must synchronise before a round advances.  "
			"0 disables explicit synchronisation."
		),
	)
	grp.add_argument(
		"--sync-interval-ms",
		type=int,
		default=500,
		help="Interval in milliseconds between synchronisation checks.",
	)
	grp.add_argument(
		"--max-drift-ms",
		type=int,
		default=0,
		help="Maximum allowed clock drift in milliseconds (0 = no limit).",
	)
