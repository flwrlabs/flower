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
"""Utilities for migrating old TOML configurations to Flower config."""


import re
from pathlib import Path
from typing import Any

import click
import typer

from .config_utils import load_and_validate, validate_federation_in_project_config
from .constant import LOCAL_SUPERLINK_ADDRESS_MAGIC_VALUE
from .flower_config import (
    init_flwr_config,
    parse_superlink_connection,
    set_default_superlink_connection,
    write_superlink_connection,
)

CONFIG_MIGRATION_NOTICE = """
#######################################################################
# CONFIGURATION MIGRATION NOTICE:
#
# What was previously called "federation config" for SuperLink
# connections in pyproject.toml has been renamed and moved.
#
# These settings are now **SuperLink connection configuration**
# and are defined in the Flower configuration file.
#
# The entries below are commented out intentionally and are kept
# only as a migration reference.
#
# Docs: https://flower.ai/docs/framework/ref-flower-configuration.html
#######################################################################

"""

CLI_NOTICE = (
    typer.style("\n🌸 Heads up from Flower!\n\n", fg=typer.colors.MAGENTA, bold=True)
    + "We detected legacy usage of this command that relies on connection\n"
    + "settings from your pyproject.toml.\n\n"
    + "Flower will migrate any relevant settings to the new Flower config.\n\n"
    + "Learn more: https://flower.ai/docs/framework/ref-flower-configuration.html\n"
)


def _is_legacy_usage(positional_arg_1: str | None, args: list[str]) -> bool:
    """Check if legacy usage is detected in the given arguments."""
    if positional_arg_1 is None:
        return False

    # If one and only one extra argument is given, assume legacy usage
    if len(args) == 1:
        return True

    # If the first positional argument looks like a path, assume legacy usage
    pth = Path(positional_arg_1)
    if pth.is_absolute() or len(pth.parts) > 1 or positional_arg_1 in (".", ".."):
        return True

    # Lastly, check if a pyproject.toml file exists at the given path
    if (pth / "pyproject.toml").exists():
        return True

    return False


def _is_migratable(app: Path) -> tuple[bool, str | None]:
    """Check if the given app path contains legacy TOML configuration.

    Parameters
    ----------
    app : Path
        Path to the Flower App.

    Returns
    -------
    tuple[bool, str | None]
        Returns (True, None) if migratable, else (False, reason).
    """
    toml_path = app / "pyproject.toml"
    if not toml_path.exists():
        return False, f"No pyproject.toml found in '{app}'"

    try:
        config, _ = load_and_validate(toml_path, check_module=False)
    except ValueError as e:
        return False, str(e)

    try:
        _ = config["tool"]["flwr"]["federations"]
        return True, None
    except KeyError:
        return False, f"No '[tool.flwr.federations]' section found in '{toml_path}'"


def _migrate_pyproject_toml_to_flower_config(
    app: Path, toml_federation: str | None
) -> tuple[list[str], str | None]:
    """Migrate old TOML configuration to Flower config."""
    # Load and validate the old TOML configuration
    toml_path = app / "pyproject.toml"
    config, _ = load_and_validate(toml_path, check_module=False)
    validate_federation_in_project_config(toml_federation, config)

    # Construct SuperLinkConnection
    toml_federations: dict[str, Any] = config["tool"]["flwr"]["federations"]
    migrated_conn_names: list[str] = []
    for name, toml_fed_config in toml_federations.items():
        if isinstance(toml_fed_config, dict):
            if "options" in toml_fed_config and "address" not in toml_fed_config:
                toml_fed_config["address"] = LOCAL_SUPERLINK_ADDRESS_MAGIC_VALUE
            # Resolve relative root-certificates path
            if cert_path := toml_fed_config.get("root-certificates"):
                if not Path(cert_path).is_absolute():
                    toml_fed_config["root-certificates"] = str(
                        (app / cert_path).expanduser().resolve()
                    )
            # Parse and write SuperLink connection
            conn = parse_superlink_connection(toml_fed_config, name)
            write_superlink_connection(conn)
            migrated_conn_names.append(name)

    # Set default federation if applicable
    default_toml_federation: str | None = toml_federations.get("default")
    if default_toml_federation in toml_federations:
        set_default_superlink_connection(default_toml_federation)

    return migrated_conn_names, default_toml_federation


def _comment_out_legacy_toml_config(app: Path) -> None:
    """Comment out legacy TOML configuration in pyproject.toml."""
    # Read pyproject.toml lines
    toml_path = app / "pyproject.toml"
    lines = toml_path.read_text(encoding="utf-8").splitlines(keepends=True)
    section_pattern = re.compile(r"\s*\[(.*)\]")

    # Comment out the [tool.flwr.federations] section
    notice_added = False
    in_federation_section = False
    with toml_path.open("w", encoding="utf-8") as f:
        for line in lines:
            # Detect section headers
            if match := section_pattern.match(line):
                section = match.group(1)
                in_federation_section = section.startswith("tool.flwr.federations")

            # Comment out lines in the federation section
            if in_federation_section:
                if not notice_added:
                    f.write(CONFIG_MIGRATION_NOTICE)
                    notice_added = True
                # Preserve empty lines and comment out others
                f.write(f"# {line}" if line.strip() != "" else line)
            else:
                f.write(line)


def migrate(
    positional_arg_1: str | None,
    args: list[str],
    ignore_legacy_usage: bool = False,
) -> None:
    """Migrate legacy TOML configuration to Flower config.

    Migrates SuperLink connection settings from `[tool.flwr.federations]` in
    pyproject.toml to the new Flower config format when legacy usage is detected
    or the migration is applicable.

    `flwr run` should call `migrate(app, [], ignore_legacy_usage=True)` to skip
    legacy usage check. Other CLI commands should call `migrate(superlink, ctx.args)`.

    Parameters
    ----------
    positional_arg_1 : str | None
        The first positional argument.
    args : list[str]
        Additional arguments. In legacy usage, this is the TOML federation name.
    ignore_legacy_usage : bool (default: False)
        Set to `True` only for `flwr run` command to skip legacy usage check.

    Raises
    ------
    click.UsageError
        If more than one extra argument is provided.
    click.ClickException
        If legacy usage detected but migration fails.

    Examples
    --------
    The following usages will trigger migration if applicable:
    - `flwr <CMD> . my-federation`
    - `flwr <CMD> ./my-app`
    - `flwr <CMD>`

    The following usages will NOT trigger migration:
    - `flwr <CMD> named-conn`

    Notes
    -----
    This function will NOT return when legacy usage is detected to force the user
    to adapt to the new usage pattern after migration.
    """
    # Initialize Flower config
    init_flwr_config()

    # Trigger the same typer error when detecting unexpected extra args
    if len(args) > 1:
        raise click.UsageError(f"Got unexpected extra arguments ({' '.join(args[1:])})")

    # Determine app path for migration
    arg1 = positional_arg_1
    app = Path(arg1) if arg1 else Path(".")
    app = app.expanduser().resolve()

    # Check if migration is applicable and if legacy usage is detected
    is_migratable, reason = _is_migratable(app)
    is_legacy = _is_legacy_usage(arg1, args) if not ignore_legacy_usage else False

    # Print notice once if legacy usage detected or migration is applicable
    if is_legacy or is_migratable:
        typer.echo(CLI_NOTICE)

    if not is_migratable:
        # Raise error if legacy usage is detected but migration is not applicable
        if is_legacy:
            raise click.ClickException(
                f"Cannot migrate configuration:\n{reason}. \nThis is expected if the "
                "migration has been previously carried out. Use `--help` after your "
                "command to see the new usage pattern."
            )
        return  # Nothing to migrate

    # Perform migration
    toml_federation = args[0] if len(args) == 1 else None
    try:
        migrated_conns, default_conn = _migrate_pyproject_toml_to_flower_config(
            app, toml_federation
        )
    except Exception as e:
        raise click.ClickException(
            f"Failed to migrate legacy TOML configuration to Flower config:\n{e!r}"
        ) from e

    typer.secho("✅ Migration completed successfully!\n", fg=typer.colors.GREEN)

    # Print migrated connections
    typer.secho("Migrated SuperLink connections:", fg=typer.colors.BLUE)
    for conn_name in migrated_conns:
        typer.secho(f"  {conn_name}", fg=typer.colors.GREEN, nl=False)
        if conn_name == default_conn:
            typer.secho(" (default)", fg=typer.colors.WHITE, nl=False)
        typer.echo()

    _comment_out_legacy_toml_config(app)

    if is_legacy:
        # print usage
        typer.secho("\nYou should now use the Flower CLI as follows:")
        ctx = click.get_current_context()
        typer.secho(ctx.get_usage() + "\n", bold=True)

        # Abort if legacy usage is detected to force user to adapt to new usage
        raise typer.Exit(code=1)


def warn_if_federation_config_overrides(
    federation_config_overrides: list[str] | None,
) -> None:
    """Warn if federation config overrides are provided.

    Parameters
    ----------
    federation_config_overrides : list[str] | None
        List of federation config override strings.
    """
    if federation_config_overrides is None:
        return

    typer.secho(
        "⚠️ Warning: `--federation-config` option is deprecated and will be ignored.\n"
        "Use Flower configuration files to set SuperLink connections.",
        fg=typer.colors.YELLOW,
        bold=True,
    )
