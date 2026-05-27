# Copyright 2026 Inria (cyrille kenfack & davide frey). All Rights Reserved.
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
import argparse
from pathlib import Path

from flwr.decentralized.common.args import _parse_args_nodes
from flwr.decentralized.simulation.args import add_simulation_args


def _parse_args_run() -> argparse.ArgumentParser:
    """Parse command-line arguments for the 'run' subcommand."""
    parser = argparse.ArgumentParser(
        description="Run a Flower Super DNode for decentralized federated learning."
    )

    _parse_args_nodes(parser)

    parser.add_argument(
        "--execution-mode",
        type=str,
        default="simulation",
        choices=["deploy", "simulation"],
        help=(
            "Execution mode for Super DNode. "
            "Use 'deploy' for real decentralized runtime. "
            "Use 'simulation' for discrete-event simulation with virtual nodes."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=500,
        help="Maximum run duration or polling timeout for the DNode runtime.",
    )
    parser.add_argument(
        "--nodeapps-pyproject",
        type=Path,
        default=Path("pyproject.toml"),
        help=(
            "Path to pyproject.toml containing [tool.flwr.app.components] "
            "entries (nodeapp*) for automatic NodeApp loading."
        ),
    )
    parser.add_argument(
        "--disable-nodeapps-autoload",
        action="store_true",
        help="Disable automatic NodeApp loading from pyproject.toml.",
    )
    parser.add_argument(
        "--node-data-config-json",
        type=str,
        default="",
        help=(
            "JSON-encoded data config to inject into all NodeApps. "
            'Example: \'{"partition-id": 2, "num-partitions": 10}\''
        ),
    )
    add_simulation_args(parser)

    return parser
