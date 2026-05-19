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
"""Flower command line interface configuration utils."""


import re
from pathlib import Path
from typing import Any, cast

import click
import tomli
import tomli_w
import typer

from flwr.cli.constant import (
    DEFAULT_FLOWER_CONFIG_TOML,
    FLOWER_CONFIG_FILE,
    SimulationBackendConfigTomlKey,
    SimulationClientResourcesTomlKey,
    SimulationInitArgsTomlKey,
    SuperLinkConnectionTomlKey,
    SuperLinkSimulationOptionsTomlKey,
)
from flwr.cli.typing import (
    SimulationBackendConfig,
    SimulationClientResources,
    SimulationInitArgs,
    SuperLinkConnection,
    SuperLinkSimulationOptions,
)
from flwr.common.config import flatten_dict
from flwr.supercore.constant import DEFAULT_SIMULATION_CONFIG
from flwr.supercore.utils import get_flwr_home


def _parse_simulation_options(options: dict[str, Any]) -> SuperLinkSimulationOptions:
    """Parse simulation options from a dictionary in a SuperLink connection."""
    num_supernodes = options.get(SuperLinkSimulationOptionsTomlKey.NUM_SUPERNODES)
    verbose = options.get(SuperLinkSimulationOptionsTomlKey.VERBOSE)
    # Validation handled in SuperLinkSimulationOptions.__post_init__

    backend_dict = options.get(SuperLinkSimulationOptionsTomlKey.BACKEND)
    simulation_backend: SimulationBackendConfig | None = None

    if isinstance(backend_dict, dict):
        # Parse client resources
        client_resources_dict = backend_dict.get(
            SimulationBackendConfigTomlKey.CLIENT_RESOURCES
        )
        client_resources: SimulationClientResources | None = None
        if isinstance(client_resources_dict, dict):
            client_resources = SimulationClientResources(
                num_cpus=client_resources_dict.get(
                    SimulationClientResourcesTomlKey.NUM_CPUS
                ),
                num_gpus=client_resources_dict.get(
                    SimulationClientResourcesTomlKey.NUM_GPUS
                ),
            )

        # Parse init args
        init_args_dict = backend_dict.get(SimulationBackendConfigTomlKey.INIT_ARGS)
        init_args: SimulationInitArgs | None = None
        if isinstance(init_args_dict, dict):
            init_args = SimulationInitArgs(
                num_cpus=init_args_dict.get(SimulationInitArgsTomlKey.NUM_CPUS),
                num_gpus=init_args_dict.get(SimulationInitArgsTomlKey.NUM_GPUS),
                logging_level=init_args_dict.get(
                    SimulationInitArgsTomlKey.LOGGING_LEVEL
                ),
                log_to_driver=init_args_dict.get(
                    SimulationInitArgsTomlKey.LOG_TO_DRIVER
                ),
            )

        simulation_backend = SimulationBackendConfig(
            name=backend_dict.get(
                SimulationBackendConfigTomlKey.NAME,
                DEFAULT_SIMULATION_CONFIG.backend,
            ),
            client_resources=client_resources,
            init_args=init_args,
        )

    # Note: validation happens in SuperLinkSimulationOptions.__post_init__
    return SuperLinkSimulationOptions(
        num_supernodes=cast(int, num_supernodes),
        backend=simulation_backend,
        verbose=verbose,
    )


def _serialize_simulation_options(
    options: SuperLinkSimulationOptions,
) -> dict[str, Any]:
    """Convert SuperLinkSimulationOptions to a dictionary for TOML serialization."""
    options_dict: dict[str, Any] = {
        SuperLinkSimulationOptionsTomlKey.NUM_SUPERNODES: options.num_supernodes
    }

    if options.verbose is not None:
        options_dict[SuperLinkSimulationOptionsTomlKey.VERBOSE] = options.verbose

    if options.backend is not None:
        backend = options.backend

        # Serialize client resources
        c_res_dict: dict[str, Any] = {}
        if backend.client_resources is not None:
            client_res = backend.client_resources
            c_res_dict = {
                SimulationClientResourcesTomlKey.NUM_CPUS: client_res.num_cpus,
                SimulationClientResourcesTomlKey.NUM_GPUS: client_res.num_gpus,
            }
            # Remove None values
            c_res_dict = {k: v for k, v in c_res_dict.items() if v is not None}

        # Serialize init args
        init_args_dict: dict[str, Any] = {}
        if backend.init_args is not None:
            init_args = backend.init_args
            init_args_dict = {
                SimulationInitArgsTomlKey.NUM_CPUS: init_args.num_cpus,
                SimulationInitArgsTomlKey.NUM_GPUS: init_args.num_gpus,
                SimulationInitArgsTomlKey.LOGGING_LEVEL: init_args.logging_level,
                SimulationInitArgsTomlKey.LOG_TO_DRIVER: init_args.log_to_driver,
            }
            # Remove None values
            init_args_dict = {k: v for k, v in init_args_dict.items() if v is not None}

        backend_dict = {
            SimulationBackendConfigTomlKey.NAME: backend.name,
            SimulationBackendConfigTomlKey.CLIENT_RESOURCES: c_res_dict,
            SimulationBackendConfigTomlKey.INIT_ARGS: init_args_dict,
        }
        # Remove empty dicts
        backend_dict = {k: v for k, v in backend_dict.items() if v}

        options_dict[SuperLinkSimulationOptionsTomlKey.BACKEND] = backend_dict

    return options_dict


def init_flwr_config() -> None:
    """Initialize the Flower configuration file."""
    config_path = get_flwr_home() / FLOWER_CONFIG_FILE

    if not config_path.exists():
        # Create parent directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        # Write Flower config file
        config_path.write_text(DEFAULT_FLOWER_CONFIG_TOML, encoding="utf-8")

        typer.secho(
            f"\nFlower configuration not found. Created default configuration"
            f" at {config_path}\n",
        )


def parse_superlink_connection(
    conn_dict: dict[str, Any], name: str
) -> SuperLinkConnection:
    """Parse SuperLink connection configuration from a TOML dictionary.

    Parameters
    ----------
    conn_dict : dict[str, Any]
        The TOML configuration dictionary for the connection.
    name : str
        The name of the connection.

    Returns
    -------
    SuperLinkConnection
        The parsed SuperLink connection configuration.
    """
    simulation_options: SuperLinkSimulationOptions | None = None
    if SuperLinkConnectionTomlKey.OPTIONS in conn_dict:
        options = conn_dict[SuperLinkConnectionTomlKey.OPTIONS]
        if isinstance(options, dict):
            simulation_options = _parse_simulation_options(options)
        else:
            raise ValueError(
                f"Invalid value for key '{SuperLinkConnectionTomlKey.OPTIONS}': "
                f"expected dict, but got {type(options).__name__}."
            )

    # Build and return SuperLinkConnection
    return SuperLinkConnection(
        name=name,
        address=conn_dict.get(SuperLinkConnectionTomlKey.ADDRESS),
        root_certificates=conn_dict.get(SuperLinkConnectionTomlKey.ROOT_CERTIFICATES),
        insecure=conn_dict.get(SuperLinkConnectionTomlKey.INSECURE),
        federation=conn_dict.get(SuperLinkConnectionTomlKey.FEDERATION),
        options=simulation_options,
    )


def serialize_superlink_connection(connection: SuperLinkConnection) -> dict[str, Any]:
    """Convert SuperLinkConnection to a dictionary for TOML serialization.

    Parameters
    ----------
    connection : SuperLinkConnection
        The SuperLink connection to serialize.

    Returns
    -------
    dict[str, Any]
        Dictionary representation suitable for TOML serialization.
    """
    # pylint: disable=protected-access
    conn_dict: dict[str, Any] = {
        SuperLinkConnectionTomlKey.ADDRESS: connection.address,
        SuperLinkConnectionTomlKey.ROOT_CERTIFICATES: connection.root_certificates,
        SuperLinkConnectionTomlKey.INSECURE: connection._insecure,
        SuperLinkConnectionTomlKey.FEDERATION: connection.federation,
    }
    # Remove None values
    conn_dict = {k: v for k, v in conn_dict.items() if v is not None}

    if connection.options is not None:
        options_dict = _serialize_simulation_options(connection.options)
        conn_dict[SuperLinkConnectionTomlKey.OPTIONS] = options_dict

    return conn_dict


def read_superlink_connection(
    connection_name: str | None = None,
) -> SuperLinkConnection:
    """Read a SuperLink connection from the Flower configuration file.

    Parameters
    ----------
    connection_name : str | None
        The name of the SuperLink connection to load. If None, the default connection
        will be loaded.

    Returns
    -------
    SuperLinkConnection
        The SuperLink connection.

    Raises
    ------
    click.ClickException
        Raised if the configuration file is corrupted, or if the requested
        connection (or default) cannot be found.
    """
    toml_dict, config_path = read_flower_config()

    try:
        superlink_config = toml_dict.get(SuperLinkConnectionTomlKey.SUPERLINK, {})

        # Load the default SuperLink connection when not provided
        if connection_name is None:
            connection_name = superlink_config.get(SuperLinkConnectionTomlKey.DEFAULT)

        # Exit when no connection name is available
        if connection_name is None:
            raise click.ClickException(
                "No SuperLink connection set. A SuperLink connection needs to be "
                "provided or one must be set as default in the Flower "
                f"configuration file ({config_path}). Specify a default SuperLink "
                "connection by adding: \n\n[superlink]\ndefault = 'connection_name'\n\n"
                f"to the Flower configuration file ({config_path})."
            )

        # Try to find the connection with the given name
        if connection_name not in superlink_config:
            msg = (
                f"SuperLink connection '{connection_name}' not found in the "
                f"Flower configuration file ({config_path})."
            )
            # If default was used, show a specific error message
            if connection_name == superlink_config.get(
                SuperLinkConnectionTomlKey.DEFAULT
            ):
                msg += (
                    f"\nPlease check that the default connection '{connection_name}' "
                    "is defined in the [superlink] section."
                )
            raise click.ClickException(msg)

        conn_dict = superlink_config[connection_name]
        return parse_superlink_connection(conn_dict, connection_name)

    except ValueError as err:
        raise click.ClickException(
            f"Failed to parse the Flower configuration file ({config_path}). {err}"
        ) from err
    except Exception as err:
        raise click.ClickException(
            f"An unexpected error occurred while reading the Flower configuration "
            f"file ({config_path}). {err}"
        ) from err


def write_superlink_connection(connection: SuperLinkConnection) -> None:
    """Write a SuperLink connection to the Flower configuration file.

    Parameters
    ----------
    connection : SuperLinkConnection
        The SuperLink connection to write to the configuration file.

    Raises
    ------
    click.ClickException
        Raised if the configuration file cannot be read or written.
    """
    toml_dict, _ = read_flower_config()

    # Ensure superlink section exists
    if SuperLinkConnectionTomlKey.SUPERLINK not in toml_dict:
        toml_dict[SuperLinkConnectionTomlKey.SUPERLINK] = {}

    superlink_config = toml_dict[SuperLinkConnectionTomlKey.SUPERLINK]

    # Serialize connection and flatten nested dicts using dotted keys
    conn_dict = serialize_superlink_connection(connection)

    # Add/update the connection
    superlink_config[connection.name] = conn_dict

    # Write back to file
    write_flower_config(toml_dict)


def set_default_superlink_connection(connection_name: str) -> None:
    """Set the default SuperLink connection."""
    toml_dict, _ = read_flower_config()

    # Get superlink section
    superlink_config = toml_dict[SuperLinkConnectionTomlKey.SUPERLINK]

    # Check if the connection exists
    if connection_name not in superlink_config:
        raise click.ClickException(
            f"SuperLink connection '{connection_name}' not found in the Flower "
            "configuration file. Cannot set as default."
        )

    # Set default connection
    superlink_config[SuperLinkConnectionTomlKey.DEFAULT] = connection_name

    # Write back to file
    write_flower_config(toml_dict)


def read_flower_config() -> tuple[dict[str, Any], Path]:
    """Read the Flower configuration file.

    Returns
    -------
    tuple[dict[str, Any], Path]
        A tuple containing the TOML configuration dictionary and the path to the
        configuration file.

    Raises
    ------
    click.ClickException
        Raised if the configuration file is corrupted.
    """
    init_flwr_config()

    config_path = get_flwr_home() / FLOWER_CONFIG_FILE

    try:
        with config_path.open("rb") as file:
            return tomli.load(file), config_path

    except tomli.TOMLDecodeError as err:
        raise click.ClickException(
            f"Failed to read the Flower configuration file ({config_path}). "
            "Please ensure it is valid TOML."
        ) from err


# This function may be subject to change once we introduce more configuration
def write_flower_config(toml_dict: dict[str, Any]) -> Path:
    """Write the Flower configuration file.

    Parameters
    ----------
    toml_dict : dict[str, Any]
        The TOML configuration dictionary to write to the file.

    Returns
    -------
    Path
        The path to the configuration file.
    """
    config_path = get_flwr_home() / FLOWER_CONFIG_FILE

    # Flatten SuperLink connections
    superlink_config: dict[str, Any] = toml_dict[SuperLinkConnectionTomlKey.SUPERLINK]
    for name in list(superlink_config.keys()):
        if isinstance(superlink_config[name], dict):
            superlink_config[name] = flatten_dict(superlink_config[name])

    # Get the standard TOML text
    toml_content = tomli_w.dumps(toml_dict)

    # Remove double quotes around multi-dot keys
    # All keys must be [A-Za-z0-9_-]+ except dots
    lines = toml_content.splitlines(keepends=True)
    pattern = re.compile(r'^"([^"]+\.[^"]+)"\s*=')
    for i, line in enumerate(lines):
        if match := pattern.match(line):
            key = match.group(1)
            lines[i] = line.replace(f'"{key}"', key)

    toml_content = "".join(lines)

    with config_path.open("w") as file:
        file.write(toml_content)

    return config_path
