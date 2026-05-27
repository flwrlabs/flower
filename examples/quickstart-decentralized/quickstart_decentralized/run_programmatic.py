# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# ==============================================================================
"""Programmatic DNode creation — no CLI, everything defined in Python.

This module shows how to build and start a ``DNode`` entirely in code,
without relying on the CLI or config files.

Usage
-----
    python -m quickstart_decentralized.run_programmatic
"""

from __future__ import annotations

import logging

from flwr.decentralized.common.node_config import _build_runtime_node
from flwr.decentralized.common.runtime_node import RuntimeNode
from flwr.decentralized.common.graph import topology_mode_dynamic
from flwr.decentralized.common.network import NetworkSettings, MdnsSettings
from flwr.decentralized.common.sampling import (
    Configuration,
    Gbps,
    PropagationPolicy,
    SelectionPolicy,
)
from flwr.decentralized.node import DNode, start_node

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def build_runtime_node() -> RuntimeNode:
    """Build a :class:`~flwr.decentralized.common.runtime_node.RuntimeNode`
    entirely in Python.

    Returns
    -------
    RuntimeNode
    """
    # 1. Sampling configuration (GBPS algorithm).
    sampling = Gbps(
        view_size=10,
        heal=2,
        swap=3,
        selection_policy=SelectionPolicy.RAND,
        propagation_policy=PropagationPolicy.PUSHPULL,
        delay=5,
        age=1,
    )
    sampling_conf = Configuration(
        config=sampling,
        config_file="/tmp/flwr_programmatic_sampling.json",
    )

    # 2. Network settings (customised mDNS idle timeout).
    network_settings = NetworkSettings(
        idle_connection_timeout_secs=60,
        enable_mdns=True,
        enable_kad=True,
        mdns=MdnsSettings(
            query_interval_secs=120,
            idle_discovery_timeout_secs=300,
        ),
    )

    # 3. Assemble the RuntimeNode directly.
    return RuntimeNode(
        context="quickstart",
        address="0.0.0.0",
        port=9100,
        topology_mode=topology_mode_dynamic(),
        sampling_conf=sampling_conf,
        network_settings=network_settings,
        tcp=True,
        udp=False,
    )


def main() -> None:
    """Build, start, and run a DNode programmatically."""
    runtime_node = build_runtime_node()

    logger.info(
        "Starting DNode | context=%s  address=%s  port=%s",
        runtime_node.context,
        runtime_node.address,
        runtime_node.port,
    )

    node = DNode(**runtime_node.to_dnode_kwargs())
    node.create_node()
    start_node(node, applications=[], timeout=600)


if __name__ == "__main__":
    main()
