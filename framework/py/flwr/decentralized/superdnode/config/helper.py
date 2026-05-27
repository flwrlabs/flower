import argparse
from pathlib import Path
from typing import Any, Sequence

from flwr.decentralized.nodeapp.node_app import NodeApp, create_nodeapps_from_pyproject


def _load_simulation_config_file(path: Path) -> dict[str, Any]:
    """Load simulation config from YAML/TOML and normalize to CLI arg keys."""
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PyYAML is required to load YAML simulation config files. "
                "Install it with: pip install pyyaml"
            ) from exc
        with open(path, encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    elif suffix == ".toml":
        try:
            import tomllib  # type: ignore[import-untyped]
        except ImportError:  # pragma: no cover
            import tomli as tomllib  # type: ignore[import-untyped,no-redef]
        with open(path, "rb") as file:
            data = tomllib.load(file)
    else:
        raise ValueError(
            f"Unsupported simulation config format '{suffix}'. "
            "Expected .yaml, .yml, or .toml."
        )

    if not isinstance(data, dict):
        return {}

    simulation = data.get("simulation", {}) if isinstance(data.get("simulation"), dict) else {}
    latency = data.get("latency", {}) if isinstance(data.get("latency"), dict) else {}
    disconnection = (
        data.get("disconnection", {})
        if isinstance(data.get("disconnection"), dict)
        else {}
    )
    synchronization = (
        data.get("synchronization", {})
        if isinstance(data.get("synchronization"), dict)
        else {}
    )
    network = data.get("network", {}) if isinstance(data.get("network"), dict) else {}
    sampling = network.get("sampling", {}) if isinstance(network.get("sampling"), dict) else {}
    topology = network.get("topology", {}) if isinstance(network.get("topology"), dict) else {}
    random_topology = topology.get("random", {}) if isinstance(topology.get("random"), dict) else {}

    flattened: dict[str, Any] = {
        "nb_nodes": simulation.get("nb_nodes", data.get("nb_nodes")),
        "sim_timeout": simulation.get("timeout", data.get("sim_timeout")),
        "max_sim_time": simulation.get("max_sim_time", data.get("max_sim_time")),
        "time_step_ms": simulation.get("time_step_ms", data.get("time_step_ms")),
        "real_time_factor": simulation.get("real_time_factor", data.get("real_time_factor")),
        "multi_thread": simulation.get("multi_thread", data.get("multi_thread")),
        "verbose_sim": simulation.get("verbose", data.get("verbose_sim")),
        "sampling_period": network.get("sampling_period", data.get("sampling_period")),
        "enable_sampling": network.get("enable_sampling", data.get("enable_sampling")),
        "sampling_config_file": sampling.get("config_file", data.get("sampling_config_file")),
        "sampling_algorithm": sampling.get("algorithm", data.get("sampling_algorithm")),
        "sampling_view_size": sampling.get("view_size", data.get("sampling_view_size")),
        "sampling_heal": sampling.get("heal", data.get("sampling_heal")),
        "sampling_swap": sampling.get("swap", data.get("sampling_swap")),
        "sampling_selection_policy": sampling.get("selection_policy", data.get("sampling_selection_policy")),
        "sampling_propagation_policy": sampling.get("propagation_policy", data.get("sampling_propagation_policy")),
        "sampling_delay": sampling.get("delay", data.get("sampling_delay")),
        "sampling_age": sampling.get("age", data.get("sampling_age")),
        "sampling_sampler_size": sampling.get("sampler_size", data.get("sampling_sampler_size")),
        "sampling_alpha": sampling.get("alpha", data.get("sampling_alpha")),
        "sampling_beta": sampling.get("beta", data.get("sampling_beta")),
        "sampling_refresh": sampling.get("refresh", data.get("sampling_refresh")),
        "topology_kind": topology.get("kind", data.get("topology_kind")),
        "topology_seed": topology.get("seed", data.get("topology_seed")),
        "random_mode": random_topology.get("mode", data.get("random_mode")),
        "random_send_to": random_topology.get("send_to", data.get("random_send_to")),
        "random_receive_from": random_topology.get("receive_from", data.get("random_receive_from")),
        "random_min_send_to": random_topology.get("min_send_to", data.get("random_min_send_to")),
        "random_max_send_to": random_topology.get("max_send_to", data.get("random_max_send_to")),
        "random_min_receive_from": random_topology.get("min_receive_from", data.get("random_min_receive_from")),
        "random_max_receive_from": random_topology.get("max_receive_from", data.get("random_max_receive_from")),
        "base_latency_ms": latency.get("base_latency_ms", data.get("base_latency_ms")),
        "jitter_factor": latency.get("jitter_factor", data.get("jitter_factor")),
        "failure_probability": disconnection.get("failure_probability", data.get("failure_probability")),
        "recovery_time": disconnection.get("recovery_time", data.get("recovery_time")),
        "sync_node_count": synchronization.get("sync_node_count", data.get("sync_node_count")),
        "sync_interval_ms": synchronization.get("sync_interval_ms", data.get("sync_interval_ms")),
        "max_drift_ms": synchronization.get("max_drift_ms", data.get("max_drift_ms")),
    }

    return {key: value for key, value in flattened.items() if value is not None}


def _apply_simulation_config_overrides(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    raw_argv: Sequence[str],
) -> argparse.Namespace:
    """Apply simulation config file values, overridden by explicit CLI flags."""
    if not getattr(args, "sim_config", None):
        return args

    config_path = Path(args.sim_config)
    if not config_path.exists():
        parser.error(f"Simulation config file not found: {config_path}")

    try:
        file_values = _load_simulation_config_file(config_path)
    except (ValueError, ImportError) as exc:
        parser.error(str(exc))

    provided_option_strings = {
        token.split("=", 1)[0]
        for token in raw_argv
        if token.startswith("-")
    }
    provided_dests = {
        action.dest
        for action in parser._actions
        if any(
            option in provided_option_strings
            for option in getattr(action, "option_strings", [])
        )
    }

    for key, value in file_values.items():
        if hasattr(args, key) and key not in provided_dests:
            setattr(args, key, value)

    return args

def _strip_superdnode_only_args(argv: Sequence[str]) -> list[str]:
    """Return argv without SuperDNode-only CLI flags.

    This keeps only arguments understood by
    :func:`flwr.decentralized.common.args.get_args_nodes`.
    """
    passthrough: list[str] = []
    skip_next = False
    for index, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue

        if token in {
            "--execution-mode",
            "--timeout",
            "--nodeapps-pyproject",
            "--disable-nodeapps-autoload",
            "--node-data-config-json",
            "--nb-nodes",
            "--sim-config",
            "--sim-timeout",
            "--max-sim-time",
            "--time-step-ms",
            "--real-time-factor",
            "--multi-thread",
            "--verbose-sim",
            "--sampling-period",
            "--enable-sampling",
            "--no-enable-sampling",
            "--network-config-mode",
            "--sampling-config-file",
            "--sampling-algorithm",
            "--sampling-view-size",
            "--sampling-heal",
            "--sampling-swap",
            "--sampling-selection-policy",
            "--sampling-propagation-policy",
            "--sampling-delay",
            "--sampling-age",
            "--sampling-sampler-size",
            "--sampling-alpha",
            "--sampling-beta",
            "--sampling-refresh",
            "--topology-kind",
            "--topology-seed",
            "--random-mode",
            "--random-send-to",
            "--random-receive-from",
            "--random-min-send-to",
            "--random-max-send-to",
            "--random-min-receive-from",
            "--random-max-receive-from",
            "--base-latency-ms",
            "--jitter-factor",
            "--failure-probability",
            "--recovery-time",
            "--sync-node-count",
            "--sync-interval-ms",
            "--max-drift-ms",
        }:
            # If the next token exists and is not an option, it belongs to this flag.
            if index + 1 < len(argv) and not argv[index + 1].startswith("-"):
                skip_next = True
            continue

        if (
            token.startswith("--execution-mode=")
            or token.startswith("--timeout=")
            or token.startswith("--nodeapps-pyproject=")
            or token.startswith("--node-data-config-json=")
            or token.startswith("--nb-nodes=")
            or token.startswith("--sim-config=")
            or token.startswith("--sim-timeout=")
            or token.startswith("--max-sim-time=")
            or token.startswith("--time-step-ms=")
            or token.startswith("--real-time-factor=")
            or token.startswith("--sampling-period=")
            or token.startswith("--network-config-mode=")
            or token.startswith("--sampling-config-file=")
            or token.startswith("--sampling-algorithm=")
            or token.startswith("--sampling-view-size=")
            or token.startswith("--sampling-heal=")
            or token.startswith("--sampling-swap=")
            or token.startswith("--sampling-selection-policy=")
            or token.startswith("--sampling-propagation-policy=")
            or token.startswith("--sampling-delay=")
            or token.startswith("--sampling-age=")
            or token.startswith("--sampling-sampler-size=")
            or token.startswith("--sampling-alpha=")
            or token.startswith("--sampling-beta=")
            or token.startswith("--sampling-refresh=")
            or token.startswith("--topology-kind=")
            or token.startswith("--topology-seed=")
            or token.startswith("--random-mode=")
            or token.startswith("--random-send-to=")
            or token.startswith("--random-receive-from=")
            or token.startswith("--random-min-send-to=")
            or token.startswith("--random-max-send-to=")
            or token.startswith("--random-min-receive-from=")
            or token.startswith("--random-max-receive-from=")
            or token.startswith("--base-latency-ms=")
            or token.startswith("--jitter-factor=")
            or token.startswith("--failure-probability=")
            or token.startswith("--recovery-time=")
            or token.startswith("--sync-node-count=")
            or token.startswith("--sync-interval-ms=")
            or token.startswith("--max-drift-ms=")
        ):
            continue

        passthrough.append(token)

    return passthrough


def _load_nodeapps_from_pyproject(pyproject_path: Path) -> list[NodeApp]:
    """Load NodeApps from pyproject if the file exists, otherwise return empty."""
    if not pyproject_path.exists():
        return []

    apps = create_nodeapps_from_pyproject(pyproject_path)
    return list(apps.values())