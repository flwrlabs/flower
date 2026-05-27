import argparse
import json
import logging
import sys
from typing import Any, Optional, Sequence

from flwr.common.logger import log
from flwr.decentralized.common.args import (
    get_args_nodes,
)
from flwr.decentralized.node import DNode
from flwr.decentralized.nodeapp import NodeApp
from flwr.decentralized.simulation.simulation import (
    build_sampling_config,
    build_sim_config,
    run_nodeapp_simulation,
)
from flwr.decentralized.superdnode.config.helper import (
    _apply_simulation_config_overrides,
    _load_nodeapps_from_pyproject,
    _strip_superdnode_only_args,
)
from flwr.decentralized.superdnode.config.parser import _parse_args_run


def _run_deploy(args: argparse.Namespace, argv: Sequence[str]) -> None:
    """Run Super DNode in deployment mode."""
    node_args = _strip_superdnode_only_args(argv)
    runtime_node = get_args_nodes(node_args)

    dnode = DNode(**runtime_node.to_dnode_kwargs())
    dnode.create_node()

    # Parse runtime data_config override from CLI
    data_config_override: dict[str, Any] = {}
    if args.node_data_config_json:
        try:
            data_config_override = json.loads(args.node_data_config_json)
            log(
                logging.INFO,
                "Parsed CLI data_config override: %s",
                data_config_override,
            )
        except json.JSONDecodeError as exc:
            log(
                logging.ERROR,
                "Failed to parse --node-data-config-json: %s",
                exc,
            )
            raise

    loaded_apps: list[NodeApp] = []
    if not args.disable_nodeapps_autoload:
        loaded_apps = _load_nodeapps_from_pyproject(args.nodeapps_pyproject)
        log(
            logging.INFO,
            "Loaded %s NodeApp(s) from %s",
            len(loaded_apps),
            args.nodeapps_pyproject,
        )
    else:
        log(logging.INFO, "NodeApp autoload disabled by CLI flag.")

    try:
        for app in loaded_apps:
            # Apply CLI data_config override if provided
            if data_config_override:
                app.set_data_config(data_config_override)
                log(
                    logging.INFO,
                    "Applied data_config override to NodeApp '%s'",
                    app.name,
                )

            app.node = dnode
            dnode.register(app_name=app.name, app=app)
            log(
                logging.INFO,
                "Registered NodeApp '%s' (subject=%s)",
                app.name,
                app.subject,
            )

        dnode.run(timeout=args.timeout)
    finally:
        for app in loaded_apps:
            dnode.unregister(app_name=app.name)
            log(logging.INFO, "Unregistered NodeApp '%s'", app.name)


def _run_simulation(args: argparse.Namespace) -> None:
    """Run Super DNode in discrete-event simulation mode."""
    loaded_apps: list[NodeApp] = []
    if not args.disable_nodeapps_autoload:
        loaded_apps = _load_nodeapps_from_pyproject(args.nodeapps_pyproject)
        log(
            logging.INFO,
            "Loaded %d NodeApp(s) from %s for simulation",
            len(loaded_apps),
            args.nodeapps_pyproject,
        )
    else:
        log(logging.INFO, "NodeApp autoload disabled by CLI flag.")

    if not loaded_apps:
        log(logging.WARNING, "No NodeApps loaded — simulation will be empty.")

    sim_config = build_sim_config(
        base_latency_ms=args.base_latency_ms,
        jitter_factor=args.jitter_factor,
        failure_probability=args.failure_probability,
        recovery_time=args.recovery_time,
        sync_node_count=args.sync_node_count,
        sync_interval_ms=args.sync_interval_ms,
        max_drift_ms=args.max_drift_ms,
        time_step_ms=args.time_step_ms,
        max_sim_time_seconds=args.max_sim_time,
        real_time_factor=args.real_time_factor,
        verbose_logging=args.verbose_sim,
    )
    network_config_mode = args.network_config_mode or (
        "sampling" if args.enable_sampling else "csr"
    )
    sampling_config = build_sampling_config(
        network_config_mode=network_config_mode,
        config_file=args.sampling_config_file,
        nb_nodes=args.nb_nodes,
        sampling_algorithm=args.sampling_algorithm,
        topology_kind=args.topology_kind,
        topology_seed=args.topology_seed,
        random_mode=args.random_mode,
        random_send_to=args.random_send_to,
        random_receive_from=args.random_receive_from,
        random_min_send_to=args.random_min_send_to,
        random_max_send_to=args.random_max_send_to,
        random_min_receive_from=args.random_min_receive_from,
        random_max_receive_from=args.random_max_receive_from,
        view_size=args.sampling_view_size,
        heal=args.sampling_heal,
        swap=args.sampling_swap,
        selection_policy=args.sampling_selection_policy,
        propagation_policy=args.sampling_propagation_policy,
        delay=args.sampling_delay,
        age=args.sampling_age,
        sampler_size=args.sampling_sampler_size,
        alpha=args.sampling_alpha,
        beta=args.sampling_beta,
        refresh=args.sampling_refresh,
        attach_sampling_to_csr=(
            args.enable_sampling and network_config_mode == "csr"
        ),
    )

    run_nodeapp_simulation(
        nb_nodes=args.nb_nodes,
        apps=loaded_apps,
        config=sim_config,
        timeout=args.sim_timeout,
        multi_thread=args.multi_thread,
        sampling_period=args.sampling_period,
        sampling_config=sampling_config,
        enable_sampling=args.enable_sampling,
    )


def run(argv: Optional[Sequence[str]] = None) -> None:
    """Internal `run` command for Flower Super DNode CLI."""
    parser = _parse_args_run()
    args = parser.parse_args(argv)
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = _apply_simulation_config_overrides(parser, args, raw_argv)

    if args.execution_mode == "simulation":
        _run_simulation(args)
        return

    _run_deploy(args, raw_argv)


def flower_super_dnode() -> None:
    """CLI entrypoint for Flower Super DNode."""
    run()
