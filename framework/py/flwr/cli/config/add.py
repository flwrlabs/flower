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
"""Flower command line interface `config add` command."""


import questionary
import typer

from flwr.cli.constant import DEFAULT_SIMULATION_BACKEND_NAME, FLOWER_CONFIG_FILE
from flwr.cli.flower_config import get_flwr_home, write_superlink_connection
from flwr.cli.typing import (
    SimulationBackendConfig,
    SimulationClientResources,
    SimulationInitArgs,
    SuperLinkConnection,
    SuperLinkSimulationOptions,
)


def _ask_superlink_connection_details() -> tuple[str | None, str | None, bool, bool]:
    """Ask for SuperLink connection details."""
    address = questionary.text("SuperLink address (e.g. 127.0.0.1:9093):").ask()
    if address:
        address = address.strip().strip('"').strip("'")

    root_certificates = questionary.path("Root certificates path (optional):").ask()
    if not root_certificates:
        root_certificates = None

    if root_certificates:
        insecure = False
    else:
        insecure = questionary.confirm("Insecure connection?", default=True).ask()

    enable_account_auth = questionary.confirm(
        "Enable account auth?", default=False
    ).ask()

    return address, root_certificates, insecure, enable_account_auth


def add() -> None:
    """Add a new SuperLink connection to the Flower configuration."""
    typer.secho("Step 1: Configure SuperLink Connection", bold=True)
    name = questionary.text("Name of the SuperLink connection:").ask()
    if not name:
        typer.secho("❌ Name is required.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    is_simulation = questionary.confirm(
        "Is this a simulation connection?", default=False
    ).ask()

    simulation_options: SuperLinkSimulationOptions | None = None
    address: str | None = None
    root_certificates: str | None = None
    insecure: bool | None = None
    enable_account_auth: bool | None = None
    federation: str | None = None

    if is_simulation:
        typer.secho("\nStep 2: Configure Simulation Options", bold=True)

        num_supernodes = questionary.text(
            "Number of SuperNodes:",
            validate=lambda text: text.isdigit() or "Please enter a valid integer",
        ).ask()

        # Ask if local or remote simulation
        is_local_simulation = questionary.confirm(
            "Is this a local simulation?", default=True
        ).ask()

        if not is_local_simulation:
            address, root_certificates, insecure, enable_account_auth = (
                _ask_superlink_connection_details()
            )

        backend_name = DEFAULT_SIMULATION_BACKEND_NAME
        client_resources: SimulationClientResources | None = None
        init_args: SimulationInitArgs | None = None

        if questionary.confirm(
            "Configure advanced simulation options?", default=False
        ).ask():
            # Client Resources & Init Args
            typer.secho("\nStep 3: Configure Client Resources & Init Args", bold=True)
            client_num_cpus = questionary.text(
                "Client num_cpus (optional, e.g. 1.0):",
                validate=lambda text: (
                    True
                    if not text
                    else (
                        text.replace(".", "", 1).isdigit()
                        or "Please enter a valid number"
                    )
                ),
            ).ask()
            client_num_gpus = questionary.text(
                "Client num_gpus (optional, e.g. 0.5):",
                validate=lambda text: (
                    True
                    if not text
                    else (
                        text.replace(".", "", 1).isdigit()
                        or "Please enter a valid number"
                    )
                ),
            ).ask()

            if client_num_cpus or client_num_gpus:
                client_resources = SimulationClientResources(
                    num_cpus=float(client_num_cpus) if client_num_cpus else None,
                    num_gpus=float(client_num_gpus) if client_num_gpus else None,
                )

            init_num_cpus = questionary.text(
                "Init num_cpus (optional, int):",
                validate=lambda text: (
                    True
                    if not text
                    else (text.isdigit() or "Please enter a valid integer")
                ),
            ).ask()
            init_num_gpus = questionary.text(
                "Init num_gpus (optional, int):",
                validate=lambda text: (
                    True
                    if not text
                    else (text.isdigit() or "Please enter a valid integer")
                ),
            ).ask()
            logging_level = questionary.text("Logging level (optional):").ask()
            log_to_drive = questionary.confirm("Log to drive? (optional)").ask()

            init_args = SimulationInitArgs(
                num_cpus=int(init_num_cpus) if init_num_cpus else None,
                num_gpus=int(init_num_gpus) if init_num_gpus else None,
                logging_level=logging_level if logging_level else None,
                log_to_drive=log_to_drive,
            )

        backend_config = SimulationBackendConfig(
            name=backend_name, client_resources=client_resources, init_args=init_args
        )

        simulation_options = SuperLinkSimulationOptions(
            num_supernodes=int(num_supernodes), backend=backend_config
        )

    else:
        # Not a simulation
        address, root_certificates, insecure, enable_account_auth = (
            _ask_superlink_connection_details()
        )

        federation = questionary.text("Federation (optional):").ask()
        if not federation:
            federation = None

    # Validate data creation by creating the object
    try:
        connection = SuperLinkConnection(
            name=name,
            address=address if address else None,
            root_certificates=root_certificates,
            insecure=insecure,
            enable_account_auth=enable_account_auth,
            federation=federation,
            options=simulation_options,
        )
    except ValueError as err:
        typer.secho(f"❌ Invalid configuration: {err}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from err

    write_superlink_connection(connection)

    # Get flowr config file
    config_path = get_flwr_home() / FLOWER_CONFIG_FILE
    typer.secho(
        f"✅ SuperLink connection '{name}' added: {config_path}", fg=typer.colors.GREEN
    )
