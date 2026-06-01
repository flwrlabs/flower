# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# ==============================================================================
"""Run a DNode from a YAML config file (static topology, auto-generated).

Usage
-----
From the ``examples/quickstart-decentralized`` directory:

    python -m quickstart_decentralized.run_static --node-name node_0

Each node in the static topology must be launched with a different
``--node-name`` value matching those defined in the topology YAML.

Override the auto-generated topology with your own file:

    python -m quickstart_decentralized.run_static \\
        --config configs/node_static.yaml \\
        --node-name node_1 \\
        --topology-file /path/to/my_topology.yaml
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from flwr.decentralized.common.args import get_args_nodes
from flwr.decentralized.node import DNode, start_node
from flwr.decentralized.nodeapp import create_nodeapps_from_pyproject

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main(argv=None) -> None:
    """Entry point — parse CLI args and start a DNode with static topology."""
    default_argv = [
        "--config", "configs/node_static.yaml",
        "--node-name", "node_1",
    ]
    argv = argv if argv is not None else (sys.argv[1:] or default_argv)

    runtime_node = get_args_nodes(argv)

    logger.info(
        "Starting DNode (static) | context=%s  address=%s  port=%s",
        runtime_node.context,
        runtime_node.address,
        runtime_node.port,
    )

    node = DNode(**runtime_node.to_dnode_kwargs())

    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    nodeapps = create_nodeapps_from_pyproject(pyproject_path)
    applications = list(nodeapps.values())
    logger.info(
        "Registering %d NodeApp(s): %s",
        len(applications),
        ", ".join(nodeapps.keys()) if nodeapps else "none",
    )

    node.create_node()
    start_node(node, applications=applications, timeout=600)


if __name__ == "__main__":
    main()
