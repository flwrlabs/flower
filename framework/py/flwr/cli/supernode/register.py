# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Flower command line interface `supernode register` command."""


from pathlib import Path
from typing import Annotated, Literal

import click
import typer
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from flwr.cli.config_migration import migrate
from flwr.cli.flower_config import read_superlink_connection
from flwr.common.constant import CliOutputFormat
from flwr.common.exit import ExitCode, flwr_exit
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    RegisterNodeRequest,
    RegisterNodeResponse,
)
from flwr.proto.control_pb2_grpc import ControlStub
from flwr.supercore.primitives.asymmetric import public_key_to_bytes, uses_nist_ec_curve

from ..utils import (
    cli_output_handler,
    flwr_cli_grpc_exc_handler,
    init_channel_from_connection,
    print_json_to_stdout,
)


def register(  # pylint: disable=R0914
    ctx: typer.Context,
    public_key: Annotated[
        Path,
        typer.Argument(
            help="Path to a P-384 (or any other NIST EC curve) public key file.",
        ),
    ],
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
) -> None:
    """Add a SuperNode to the federation."""
    # Load public key
    public_key_bytes = try_load_public_key(public_key.expanduser())

    with cli_output_handler(output_format=output_format) as is_json:
        # Migrate legacy usage if any
        migrate(superlink, args=ctx.args)

        # Read superlink connection configuration
        superlink_connection = read_superlink_connection(superlink)
        channel = None

        try:
            channel = init_channel_from_connection(superlink_connection)
            stub = ControlStub(channel)

            _register_node(
                stub=stub,
                public_key=public_key_bytes,
                is_json=is_json,
            )

        finally:
            if channel:
                channel.close()


def _register_node(stub: ControlStub, public_key: bytes, is_json: bool) -> None:
    """Register a node."""
    with flwr_cli_grpc_exc_handler():
        response: RegisterNodeResponse = stub.RegisterNode(
            request=RegisterNodeRequest(public_key=public_key)
        )
    if response.node_id:
        typer.secho(
            f"✅ SuperNode {response.node_id} registered successfully.",
            fg=typer.colors.GREEN,
        )
        if is_json:
            print_json_to_stdout(
                {
                    "success": True,
                    "node-id": response.node_id,
                }
            )
    else:
        raise click.ClickException("SuperNode couldn't be registered.")


def try_load_public_key(public_key_path: Path) -> bytes:
    """Try to load a public key from a file."""
    if not public_key_path.exists():
        raise click.ClickException(
            f"Public key file '{public_key_path}' does not exist."
        )

    with open(public_key_path, "rb") as key_file:
        try:
            public_key = serialization.load_ssh_public_key(key_file.read())

            if not isinstance(public_key, ec.EllipticCurvePublicKey):
                raise ValueError(f"Not an EC public key, got {type(public_key)}")

            # Verify it's one of the approved NIST curves
            if not uses_nist_ec_curve(public_key):
                raise ValueError(
                    f"EC curve {public_key.curve.name} is not an approved NIST curve"
                )

        except (ValueError, UnsupportedAlgorithm) as err:
            flwr_exit(
                ExitCode.FLWRCLI_NODE_AUTH_PUBLIC_KEY_INVALID,
                str(err),
            )
    return public_key_to_bytes(public_key)
