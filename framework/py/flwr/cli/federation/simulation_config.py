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
"""Flower command line interface `federation simulation-config` command."""


from typing import Annotated, Any, Literal

import typer

from flwr.cli.utils import (
    cli_output_control_stub,
    flwr_cli_grpc_exc_handler,
    print_json_to_stdout,
)
from flwr.common.constant import CliOutputFormat
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ConfigureSimulationFederationRequest,
    ConfigureSimulationFederationResponse,
)
from flwr.proto.control_pb2_grpc import ControlStub
from flwr.proto.federation_config_pb2 import SimulationConfig  # pylint: disable=E0611
from flwr.supercore.constant import DEFAULT_SIMULATION_CONFIG

from .error_handlers import handle_invite_grpc_error


def _handle_none_field(cfg: SimulationConfig, field_name: str) -> Any:
    """Convert unset optional fields to None."""
    if not cfg.HasField(field_name):  # type: ignore
        return None
    return getattr(cfg, field_name)


def simulation_config(  # pylint: disable=R0913,R0917,W0613
    federation: Annotated[
        str | None,
        typer.Argument(
            help="Name of the federation; must be in the "
            "format `@<account>/<federation>`."
        ),
    ] = None,
    superlink: Annotated[
        str | None,
        typer.Argument(help="Name of the SuperLink connection."),
    ] = None,
    output_format: Annotated[
        Literal["default", "json"],
        typer.Option(
            "--format",
            case_sensitive=False,
            help="Format output using 'default' view or 'json'",
        ),
    ] = CliOutputFormat.DEFAULT,
    num_supernodes: Annotated[
        int,
        typer.Option(
            "--num-supernodes",
            help="The number of virtual SuperNodes in the simulation",
            min=1,
        ),
    ] = DEFAULT_SIMULATION_CONFIG.num_supernodes,
    client_resources_num_cpus: Annotated[
        int,
        typer.Option(
            "--client-resources-num-cpus",
            help="CPUs assigned to the execution of each ClientApp",
            min=1,
        ),
    ] = DEFAULT_SIMULATION_CONFIG.client_resources_num_cpus,
    client_resources_num_gpus: Annotated[
        float,
        typer.Option(
            "--client-resources-num-gpus",
            help="Ratio of a GPU VRAM assigned to the execution of each ClientApp",
            min=0.0,
        ),
    ] = DEFAULT_SIMULATION_CONFIG.client_resources_num_gpus,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Run the Simulation Runtime with verbose logs",
        ),
    ] = DEFAULT_SIMULATION_CONFIG.verbose,
    backend: Annotated[
        Literal["ray"],
        typer.Option(
            "--backend-name",
            case_sensitive=False,
            help="Choice of backend name (Currently, only 'ray' is supported).",
        ),
    ] = DEFAULT_SIMULATION_CONFIG.backend,  # type: ignore
    init_args_num_cpus: Annotated[
        int | None,
        typer.Option(
            "--init-args-num-cpus",
            help="Number of CPUs to make available to the Simulation Runtime.",
            min=1,
        ),
    ] = _handle_none_field(DEFAULT_SIMULATION_CONFIG, "init_args_num_cpus"),
    init_args_num_gpus: Annotated[
        int | None,
        typer.Option(
            "--init-args-num-gpus",
            help="Number of GPUs to make available to the Simulation Runtime",
            min=0,
        ),
    ] = _handle_none_field(DEFAULT_SIMULATION_CONFIG, "init_args_num_gpus"),
    init_args_logging_level: Annotated[
        str | None,
        typer.Option(
            "--init-args-logging-level",
            help="Control logging level in Simulation Runtime.",
        ),
    ] = DEFAULT_SIMULATION_CONFIG.init_args_logging_level,
    init_args_log_to_driver: Annotated[
        bool,
        typer.Option(
            "--init-args-log-to-driver",
            help="Whether to propagate logs from Simulation Runtime upwards.",
        ),
    ] = DEFAULT_SIMULATION_CONFIG.init_args_log_to_driver,
) -> None:
    """Configure a Federation using the Simulation Runtime."""
    with cli_output_control_stub(superlink, output_format) as (stub, is_json):
        request = ConfigureSimulationFederationRequest(
            federation_name=federation or "",
            config=SimulationConfig(
                num_supernodes=num_supernodes,
                client_resources_num_cpus=client_resources_num_cpus,
                client_resources_num_gpus=client_resources_num_gpus,
                backend=backend,
                verbose=verbose,
                init_args_num_cpus=init_args_num_cpus,
                init_args_num_gpus=init_args_num_gpus,
                init_args_logging_level=init_args_logging_level,
                init_args_log_to_driver=init_args_log_to_driver,
            ),
        )
        _ = is_json
        _configure_federation_for_simulation(
            stub=stub,
            request=request,
            is_json=is_json,
        )


def _configure_federation_for_simulation(
    stub: ControlStub,
    request: ConfigureSimulationFederationRequest,
    is_json: bool,
) -> None:
    """Send a request to configure a federation for simulation."""
    with flwr_cli_grpc_exc_handler(handle_invite_grpc_error):
        _: ConfigureSimulationFederationResponse = stub.ConfigureSimulationFederation(
            request
        )

    if is_json:
        print_json_to_stdout({"success": True})
    else:
        typer.secho("✅ Updated simulation configuration.")
