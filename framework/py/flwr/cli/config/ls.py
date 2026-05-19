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
"""Flower command line interface `config list` command."""


from typing import Annotated, Literal

import typer

from flwr.common.constant import CliOutputFormat

from ..constant import SuperLinkConnectionTomlKey
from ..flower_config import read_flower_config
from ..utils import cli_output_handler, print_json_to_stdout


def ls(
    output_format: Annotated[
        Literal["default", "json"],
        typer.Option(
            "--format",
            case_sensitive=False,
            help="Format output using 'default' view or 'json'",
        ),
    ] = CliOutputFormat.DEFAULT,
) -> None:
    """List all SuperLink connections (alias: ls)."""
    with cli_output_handler(output_format=output_format) as is_json:
        # Load Flower Config
        config, config_path = read_flower_config()

        # Get `superlink` tables
        superlink_connections = config.get(SuperLinkConnectionTomlKey.SUPERLINK, {})

        # Get default, then pop from dict
        default = superlink_connections.pop(SuperLinkConnectionTomlKey.DEFAULT, None)

        connection_names = list(superlink_connections.keys())

        if is_json:
            conn = {
                SuperLinkConnectionTomlKey.SUPERLINK: connection_names,
                SuperLinkConnectionTomlKey.DEFAULT: default,
            }
            print_json_to_stdout(conn)
        else:
            typer.secho("Flower Config file: ", fg=typer.colors.BLUE, nl=False)
            typer.secho(f"{config_path}", fg=typer.colors.GREEN)
            typer.secho("SuperLink connections:", fg=typer.colors.BLUE)
            # List SuperLink connections and highlight default
            for k in connection_names:
                typer.secho(f"  {k}", fg=typer.colors.GREEN, nl=False)
                if k == default:
                    typer.secho(
                        f" ({SuperLinkConnectionTomlKey.DEFAULT})",
                        fg=typer.colors.WHITE,
                        nl=False,
                    )
                typer.echo()
