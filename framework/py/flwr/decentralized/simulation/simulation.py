"""Simulation entrypoint for Super DNode.

This module bridges :class:`~flwr.decentralized.nodeapp.NodeApp` prototypes with
the nodemanager discrete-event simulation engine.  The main function,
:func:`run_nodeapp_simulation`, clones each prototype ``nb_nodes`` times via
:meth:`~flwr.decentralized.nodeapp.NodeApp.for_node` and hands the resulting
instances to ``run_discrete_event_simulation``.
"""

import logging
from typing import Dict, List, Optional

from nodemanager.simulation import run_discrete_event_simulation  # type: ignore[import-untyped]

from flwr.common.logger import log
from flwr.decentralized.common.graph import RandomExact, RandomRange, generate_simulation_csr
from flwr.decentralized.common.sampling import (
	Basalt,
	Brahams,
	Configuration,
	Gbps,
	PropagationPolicy,
	SelectionPolicy,
)
from flwr.decentralized.simulation.config import (
	DiscreteEventSimConfig,
	DisconnectionConfig,
	LatencyConfig,
	SynchronizationConfig,
)

# NodeApp is imported lazily inside the function to avoid circular imports.


def build_sim_config(
	*,
	base_latency_ms: float = 50.0,
	jitter_factor: float = 0.1,
	failure_probability: float = 0.0,
	recovery_time: float = 10.0,
	sync_node_count: int = 0,
	sync_interval_ms: int = 500,
	max_drift_ms: int = 0,
	time_step_ms: int = 100,
	max_sim_time_seconds: float = 3600.0,
	real_time_factor: float = 1.0,
	verbose_logging: bool = False,
) -> DiscreteEventSimConfig:
	"""Construct a :class:`DiscreteEventSimConfig` from plain scalar arguments.

	This is a convenience wrapper so CLI code does not need to import the
	nested config dataclasses directly.
	"""
	return DiscreteEventSimConfig(
		latency_config=LatencyConfig(
			base_latency_ms=base_latency_ms,
			jitter_factor=jitter_factor,
		),
		disconnection_config=DisconnectionConfig(
			failure_probability=failure_probability,
			recovery_time_seconds=recovery_time,
		),
		sync_config=SynchronizationConfig(
			sync_node_count=sync_node_count,
			sync_interval_ms=sync_interval_ms,
			max_drift_ms=max_drift_ms,
		),
		time_step_ms=time_step_ms,
		max_sim_time_seconds=max_sim_time_seconds,
		real_time_factor=real_time_factor,
		verbose_logging=verbose_logging,
	)


def build_sampling_config(
	*,
	network_config_mode: str = "sampling",
	config_file: str = "config_sampling.json",
	nb_nodes: Optional[int] = None,
	sampling_algorithm: str = "gbps",
	topology_kind: str = "ring",
	topology_seed: int = 42,
	random_mode: str = "exact",
	random_send_to: int = 1,
	random_receive_from: int = 1,
	random_min_send_to: Optional[int] = None,
	random_max_send_to: int = 2,
	random_min_receive_from: Optional[int] = None,
	random_max_receive_from: int = 2,
	view_size: int = 4,
	heal: int = 0,
	swap: int = 0,
	selection_policy: str = "old",
	propagation_policy: str = "pushpull",
	delay: int = 2,
	age: int = 1,
	sampler_size: int = 8,
	alpha: float = 0.5,
	beta: float = 0.5,
	refresh: int = 1,
	attach_sampling_to_csr: bool = False,
) -> Configuration:
	"""Construct a valid nodemanager sampling/network configuration.

	This ensures simulation always has an explicit network configuration file,
	which is required by nodemanager.
	"""
	if sampling_algorithm == "gbps":
		sampling = Gbps(
			view_size=view_size,
			heal=heal,
			swap=swap,
			selection_policy=SelectionPolicy(selection_policy),
			propagation_policy=PropagationPolicy(propagation_policy),
			delay=delay,
			age=age,
		)
	elif sampling_algorithm == "brahams":
		sampling = Brahams(
			view_size=view_size,
			sampler_size=sampler_size,
			alpha=alpha,
			beta=beta,
			delay=delay,
		)
	elif sampling_algorithm == "basalt":
		sampling = Basalt(
			view_size=view_size,
			refresh=refresh,
			delay=delay,
		)
	else:
		raise ValueError(
			"Unsupported sampling_algorithm. Expected one of: gbps, brahams, basalt"
		)

	if network_config_mode == "sampling":
		return Configuration(config=sampling, config_file=config_file)

	if nb_nodes is None:
		raise ValueError("nb_nodes is required when network_config_mode='csr'")

	random_config = None
	if topology_kind == "random":
		if random_mode == "range":
			random_config = RandomRange(
				min_send_to=random_min_send_to,
				max_send_to=random_max_send_to,
				min_receive_from=random_min_receive_from,
				max_receive_from=random_max_receive_from,
			)
		else:
			random_config = RandomExact(
				send_to=random_send_to,
				receive_from=random_receive_from,
			)

	csr, _ = generate_simulation_csr(
		node_count=nb_nodes,
		kind=topology_kind,
		random=random_config,
		sampling=sampling if attach_sampling_to_csr else None,
		seed=topology_seed,
	)

	return Configuration(config=csr, config_file=config_file)


def run_nodeapp_simulation(
	nb_nodes: int,
	apps: "List",  # List[NodeApp]
	config: DiscreteEventSimConfig,
	*,
	timeout: int = 300,
	multi_thread: bool = False,
	sampling_period: int = 1000,
	sampling_config: Optional[Configuration] = None,
	enable_sampling: bool = False,
) -> None:
	"""Run a discrete-event simulation with virtual ``NodeApp`` instances.

	For each ``NodeApp`` prototype in *apps* the function creates *nb_nodes*
	independent copies via :meth:`~flwr.decentralized.nodeapp.NodeApp.for_node`
	and registers them under the app's subject name in the
	``applications`` dict expected by ``run_discrete_event_simulation``.

	Parameters
	----------
	nb_nodes : int
		Number of virtual nodes per app subject.
	apps : list[NodeApp]
		Prototype NodeApp instances (one per subject).  Each app must **not**
		already have a fully configured ``data_config`` — partition-id and
		num-partitions will be injected per clone.
	config : DiscreteEventSimConfig
		Network / timing configuration for the simulation engine.
	timeout : int
		Wall-clock timeout in seconds.
	multi_thread : bool
		When ``True`` each virtual node runs in its own thread.
	sampling_period : int
		Peer-sampling period in milliseconds.
	sampling_config : Configuration | None
		Gossip/peer-sampling configuration.  ``None`` uses the engine default.
	enable_sampling : bool
		Whether to activate peer sampling.
	"""
	applications: Dict[str, List] = {}

	for app in apps:
		instances = [app.for_node(i, nb_nodes) for i in range(nb_nodes)]
		applications[app.subject] = instances
		log(
			logging.INFO,
			"Prepared %d instance(s) for subject '%s'",
			nb_nodes,
			app.subject,
		)

	effective_sampling_config = sampling_config

	log(
		logging.INFO,
		"Starting discrete-event simulation: nb_nodes=%d, subjects=%s, "
		"max_sim_time=%.1fs, timeout=%ds",
		nb_nodes,
		list(applications.keys()),
		config.max_sim_time_seconds,
		timeout,
	)

	run_discrete_event_simulation(
		nb_nodes=nb_nodes,
		applications=applications,
		sampling_config=effective_sampling_config,
		sampling_period=sampling_period,
		config=config,
		timeout=timeout,
		enable_sampling=enable_sampling,
		multi_thread=multi_thread,
	)



