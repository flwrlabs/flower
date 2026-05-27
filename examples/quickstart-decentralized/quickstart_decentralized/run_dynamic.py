# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Run a DNode from a YAML config file (dynamic topology with GBPS sampling).

Usage
-----
From the ``examples/quickstart-decentralized`` directory:

    python -m quickstart_decentralized.run_dynamic

Override any value at the CLI — CLI flags always take priority over the file:

    python -m quickstart_decentralized.run_dynamic \\
        --config configs/node_dynamic.yaml \\
        --port 9200 \\
        --bootnodes 192.168.1.5:9100

"""

from __future__ import annotations

import logging
import sys

from flwr.decentralized.common.args import get_args_nodes
from flwr.decentralized.node import DNode, start_node

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main(argv=None) -> None:
    """Entry point — parse CLI args and start a DNode with a dummy app."""
    # Provide sensible defaults so the example works without extra flags.
    default_argv = [
        "--config", "configs/node_dynamic.yaml",
    ]
    argv = argv if argv is not None else (sys.argv[1:] or default_argv)

    # Parse CLI arguments (merged with config file when --config is given).
    runtime_node = get_args_nodes(argv)

    logger.info(
        "Starting DNode | context=%s  address=%s  port=%s  topology=%s",
        runtime_node.context,
        runtime_node.address,
        runtime_node.port,
        runtime_node.topology_mode,
    )

    # Build the DNode from the resolved RuntimeNode.
    node = DNode(**runtime_node.to_dnode_kwargs())

    # Start the node runtime (connects to the network).
    node.create_node()

    # In a real application you would register one or more App instances:
    #
    #   from my_app import MyFLApp
    #   app = MyFLApp(name="fl_app")
    #   start_node(node, applications=[app], timeout=600)
    #
    # Here we just run the bare node loop for demonstration.
    start_node(node, applications=[], timeout=600)


if __name__ == "__main__":
    main()
