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
"""CLI argument helpers for decentralized topology and node configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

from flwr.decentralized.common.typing import DEFAULT_IP_ADDRESS, DEFAULT_PORT

if TYPE_CHECKING:
    from flwr.decentralized.common.runtime_node import RuntimeNode


def add_args_topology(parser: argparse.ArgumentParser) -> None:
    """Add command-line arguments for decentralized topology configuration."""
    parser.add_argument(
        "--topology-mode",
        type=str,
        default="dynamic",
        choices=["dynamic", "static"],
        help="Mode of the decentralized topology.",
    )

    parser.add_argument(
        "--topology-file",
        type=Path,
        default=None,
        help=(
            "Path to a YAML file defining the decentralized topology. "
            "Required if --topology-mode is set to 'static'."
        ),
    )


def validate_topology_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    """Validate topology-related command-line arguments.
    
    When a config file is provided, topology file validation is deferred since
    topology configuration may be loaded from the config and merged with CLI
    overrides. Validation only applies when using CLI args exclusively (no config).
    """
    if (
        args.topology_mode == "static"
        and args.topology_file is None
        and args.config is None
    ):
        parser.error(
            "--topology-file is required when --topology-mode is set to 'static' "
            "(or provide a --config file that defines topology.file or topology.generate)."
        )


# ---------------------------------------------------------------------------
# DNode argument parser
# ---------------------------------------------------------------------------


def _parse_args_nodes(parser: argparse.ArgumentParser) -> None:
    """Return an :class:`~argparse.ArgumentParser` for DNode creation.

    Arguments defined here can be used standalone or merged with a
    ``--config`` YAML / TOML file (CLI values always take precedence).

    Returns
    -------
    argparse.ArgumentParser
    """

    # --- Config file (optional, all other args become overrides) ---
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "Path to a YAML (.yaml / .yml) or TOML (.toml) configuration "
            "file. When provided, all other CLI arguments override the "
            "corresponding values in the file."
        ),
    )

    # --- Identity ---
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help=(
            "Runtime context name for this node. Only nodes sharing the "
            "same context can discover each other and communicate."
        ),
    )
    parser.add_argument(
        "--address",
        type=str,
        default=DEFAULT_IP_ADDRESS,
        help=f"Host / IP address for this node (default: {DEFAULT_IP_ADDRESS}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Listening port for this node (default: {DEFAULT_PORT}).",
    )

    # --- Transport ---
    parser.add_argument(
        "--tcp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TCP transport (default: true).",
    )
    parser.add_argument(
        "--udp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable UDP transport (default: false).",
    )

    # --- Bootstrap ---
    parser.add_argument(
        "--bootnodes",
        type=str,
        nargs="*",
        default=None,
        metavar="ADDR",
        help=(
            "Space-separated list of bootstrap peer addresses, e.g. "
            "127.0.0.1:9001 127.0.0.1:9002."
        ),
    )

    # --- Topology (reuse shared adders) ---
    add_args_topology(parser)

    # Additional static-topology helpers
    parser.add_argument(
        "--node-name",
        type=str,
        default=None,
        help=(
            "Name identifying this node within the static topology YAML. "
            "Required when --topology-mode is 'static'."
        ),
    )


def validate_node_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """Validate parsed DNode arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from :func:`_parse_args_nodes`.
    parser : argparse.ArgumentParser
        Parser used for error reporting.

    Raises
    ------
    SystemExit
        If a required argument combination is invalid.
    """
    validate_topology_args(args, parser)

    if args.config is None:
        # context is mandatory when not using a config file
        if not args.context:
            parser.error(
                "--context is required (or provide a --config file that " "defines it)."
            )
        # static mode also needs a node-name when no config file
        if (
            getattr(args, "topology_mode", None) == "static"
            and getattr(args, "node_name", None) is None
        ):
            parser.error("--node-name is required when --topology-mode is 'static'.")


def get_args_nodes(
    argv: Optional[Sequence[str]] = None,
) -> RuntimeNode:
    """Parse CLI arguments and return a fully resolved
    :class:`~flwr.decentralized.common.runtime_node.RuntimeNode`.

    If ``--config`` is provided, values from the file are used as base and
    any CLI flag overrides them. When ``--config`` is absent, all parameters
    must be supplied via CLI flags (or defaults).

    Parameters
    ----------
    argv : sequence of str | None
        Argument list to parse. ``None`` reads from ``sys.argv``.

    Returns
    -------
    RuntimeNode
        A validated, ready-to-use configuration container for ``DNode``.

    Examples
    --------
    From the command line::

        flwr-super-dnode run --context cls --address 0.0.0.0 --port 9100 \\
            --topology-mode dynamic

    With a config file::

        flwr-super-dnode run --config node.yaml --port 9200  # port overrides file

    Programmatically::

        >>> runtime_node = get_args_nodes(
        ...     ["--context", "cls", "--address", "0.0.0.0", "--port", "9100",
        ...      "--topology-mode", "dynamic"]
        ... )
        >>> dnode = DNode(**runtime_node.to_dnode_kwargs())
    """
    # Import here to avoid circular imports at module level.
    from flwr.decentralized.common.node_config import (
        _build_runtime_node,
        load_node_config_toml,
        load_node_config_yaml,
    )

    parser = argparse.ArgumentParser()
    _parse_args_nodes(parser)
    args = parser.parse_args(argv)

    # Track which options were explicitly provided on the CLI.
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    provided_option_strings = {
        token.split("=", 1)[0] for token in raw_argv if token.startswith("-")
    }
    provided_dests = {
        action.dest
        for action in parser._actions
        if any(
            option in provided_option_strings
            for option in getattr(action, "option_strings", [])
        )
    }

    validate_node_args(args, parser)

    # Build overrides using only arguments explicitly set on CLI.
    overrides = {
        k: v
        for k, v in vars(args).items()
        if k in provided_dests and v is not None and k != "config"
    }

    if args.config is not None:
        config_path = Path(args.config)
        suffix = config_path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            return load_node_config_yaml(config_path, overrides=overrides)
        elif suffix == ".toml":
            return load_node_config_toml(config_path, overrides=overrides)
        else:
            parser.error(
                f"Unsupported config file format '{suffix}'. "
                "Expected .yaml, .yml, or .toml."
            )

    # No config file: build entirely from CLI args.
    return _build_runtime_node(data={}, overrides=overrides)
