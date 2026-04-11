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
"""Flower command line interface `run` command."""


import hashlib
from pathlib import Path
from typing import Annotated, Any, Literal

import click
import typer

from flwr.cli.build import build_fab_from_disk, get_fab_filename
from flwr.cli.config_migration import migrate
from flwr.cli.config_utils import load_and_validate
from flwr.cli.constant import FEDERATION_CONFIG_HELP_MESSAGE, RUN_CONFIG_HELP_MESSAGE
from flwr.cli.flower_config import read_superlink_connection
from flwr.cli.typing import SuperLinkConnection
from flwr.common.config import get_metadata_from_config, parse_config_args
from flwr.common.constant import FAB_CONFIG_FILE, CliOutputFormat
from flwr.common.serde import fab_to_proto, user_config_to_proto
from flwr.common.typing import Fab
from flwr.proto.control_pb2 import StartRunRequest  # pylint: disable=E0611
from flwr.proto.control_pb2_grpc import ControlStub
from flwr.proto.federation_config_pb2 import SimulationConfig  # pylint: disable=E0611
from flwr.supercore.utils import (
    check_federation_format,
    parse_app_spec,
    simulation_config_from_json,
)

from ..log import start_stream
from ..utils import (
    cli_output_handler,
    flwr_cli_grpc_exc_handler,
    init_channel_from_connection,
    print_json_to_stdout,
)

CONN_REFRESH_PERIOD = 60  # Connection refresh period for log streaming (seconds)


# pylint: disable-next=too-many-locals, too-many-branches, R0913, R0917
def run(
    app: Annotated[
        str,
        typer.Argument(
            help="Path of the Flower App to run, or a remote app spec "
            "like '@account_name/app_name'."
        ),
    ] = ".",
    superlink: Annotated[
        str | None,
        typer.Argument(help="Name of the SuperLink connection."),
    ] = None,
    federation: Annotated[
        str | None,
        typer.Option(
            "--federation",
            help="The federation to submit the run to; must be in the "
            "format `@<account>/<federation>`.",
        ),
    ] = None,
    run_config_overrides: Annotated[
        list[str] | None,
        typer.Option(
            "--run-config",
            "-c",
            help=RUN_CONFIG_HELP_MESSAGE,
        ),
    ] = None,
    federation_config_overrides: Annotated[
        list[str] | None,
        typer.Option(
            "--federation-config",
            help=FEDERATION_CONFIG_HELP_MESSAGE,
        ),
    ] = None,
    stream: Annotated[
        bool,
        typer.Option(
            "--stream",
            help="Use `--stream` with `flwr run` to display logs;\n "
            "logs are not streamed by default.",
        ),
    ] = False,
    output_format: Annotated[
        Literal["default", "json"],
        typer.Option(
            "--format",
            case_sensitive=False,
            help="Format output using 'default' view or 'json'",
        ),
    ] = CliOutputFormat.DEFAULT,
) -> None:
    """Run Flower App."""
    with cli_output_handler(output_format=output_format) as is_json:
        # Determine if app is remote
        app_spec = None
        config: dict[str, Any] = {}
        if app.startswith("@"):
            # Validate app version and ID format
            try:
                _ = parse_app_spec(app)
            except ValueError as e:
                raise click.ClickException(str(e)) from e

            app_path = Path(".")
            app_spec = app

        # Validate TOML configuration for local app
        else:
            app_path = Path(app).expanduser().resolve()  # Resolve path to absolute

            # Migrate legacy usage if any
            migrate(str(app_path), [], ignore_legacy_usage=True)

            config, warnings = load_and_validate(app_path / FAB_CONFIG_FILE)
            if warnings:
                warning_path = app_path / FAB_CONFIG_FILE
                typer.secho(
                    f"Flower App configuration warnings in '{warning_path}':\n"
                    + "\n".join([f"- {line}" for line in warnings]),
                    fg=typer.colors.YELLOW,
                    bold=True,
                )

        # Read superlink connection configuration
        superlink_connection = read_superlink_connection(superlink)

        _run_with_control_api(
            app_path,
            config,
            federation,
            superlink_connection,
            run_config_overrides,
            federation_config_overrides,
            stream,
            is_json,
            app_spec,
        )


# pylint: disable-next=R0912, R0913, R0914, R0917
def _run_with_control_api(
    app: Path,
    config: dict[str, Any],
    federation: str | None,
    superlink_connection: SuperLinkConnection,
    config_overrides: list[str] | None,
    federation_config_overrides: list[str] | None,
    stream: bool,
    is_json: bool,
    app_spec: str | None,
) -> None:
    channel = None
    is_remote_app = app_spec is not None

    # Determine federation to use
    if federation:  # Override federation from CLI
        check_federation_format(federation)
    else:  # Use federation from SuperLink connection if set
        federation = superlink_connection.federation or ""

    try:
        channel = init_channel_from_connection(superlink_connection)
        stub = ControlStub(channel)

        # Build FAB if local app
        if not is_remote_app:
            fab_bytes = build_fab_from_disk(app)
            fab_hash = hashlib.sha256(fab_bytes).hexdigest()
            fab_id, fab_version = get_metadata_from_config(config)
            fab = Fab(fab_hash, fab_bytes, {})
        # Skip FAB build if remote app
        else:
            # Use empty values for FAB
            fab_id = fab_version = fab_hash = ""
            fab = Fab(fab_hash, b"", {})

        req = StartRunRequest(
            fab=fab_to_proto(fab),
            override_config=user_config_to_proto(parse_config_args(config_overrides)),
            federation=federation,
            override_federation_config=_parse_federation_config_overrides(
                federation_config_overrides, superlink_connection
            ),
            app_spec=app_spec or "",
        )
        with flwr_cli_grpc_exc_handler():
            res = stub.StartRun(req)

        if res.HasField("run_id"):
            typer.secho(
                f"🎊 Successfully started run {res.run_id}", fg=typer.colors.GREEN
            )
        else:
            raise click.ClickException("Failed to start run")

        if is_json:
            # Only include FAB metadata if we actually built a local FAB
            payload: dict[str, Any] = {
                "success": res.HasField("run_id"),
                "run-id": f"{res.run_id}" if res.HasField("run_id") else None,
            }
            if not is_remote_app:
                payload.update(
                    {
                        "fab-id": fab_id,
                        "fab-name": fab_id.rsplit("/", maxsplit=1)[-1],
                        "fab-version": fab_version,
                        "fab-hash": fab_hash[:8],
                        "fab-filename": get_fab_filename(config, fab_hash),
                    }
                )
            print_json_to_stdout(payload)

        if stream:
            start_stream(res.run_id, channel, CONN_REFRESH_PERIOD)
    finally:
        if channel:
            channel.close()


def _parse_federation_config_overrides(
    federation_config_overrides: list[str] | None,
    superlink_connection: SuperLinkConnection,
) -> SimulationConfig | None:
    override_federation_config = _parse_deprecated_options(superlink_connection)
    if federation_config_overrides:
        if override_federation_config:
            typer.secho(
                "⚠️ Warning: `--federation-config` was provided, so deprecated "
                "`options.` entries from the SuperLink connection in your Flower "
                "configuration will be ignored.",
                fg=typer.colors.YELLOW,
            )

        # Convert CLI overrides to `SimulationConfig`
        # The `-` in CLI keys shall be replaced with `_` to match proto field names
        tmp_dict = parse_config_args(federation_config_overrides)
        tmp_dict = {k.replace("-", "_"): v for k, v in tmp_dict.items()}
        override_federation_config = simulation_config_from_json(tmp_dict)

    return override_federation_config


def _parse_deprecated_options(conn: SuperLinkConnection) -> SimulationConfig | None:
    if not (opt := conn.options):
        return None
    typer.secho(
        "⚠️ Warning: `options.` fields in the SuperLink connection configuration are "
        "deprecated. Use `--federation-config` with `flwr run` instead. "
        "Alternatively, permanently set your simulation configuration via "
        "`flwr federation simulation-config`.",
        fg=typer.colors.YELLOW,
    )
    kwargs: dict[str, Any] = {
        "num_supernodes": opt.num_supernodes,
        "verbose": opt.verbose,
    }
    if opt.backend:
        kwargs["backend"] = opt.backend.name
        if opt.backend.client_resources:
            kwargs["client_resources_num_cpus"] = opt.backend.client_resources.num_cpus
            kwargs["client_resources_num_gpus"] = opt.backend.client_resources.num_gpus
        if opt.backend.init_args:
            kwargs["init_args_num_cpus"] = opt.backend.init_args.num_cpus
            kwargs["init_args_num_gpus"] = opt.backend.init_args.num_gpus
            kwargs["init_args_logging_level"] = opt.backend.init_args.logging_level
            kwargs["init_args_log_to_driver"] = opt.backend.init_args.log_to_driver
    return SimulationConfig(**kwargs)
