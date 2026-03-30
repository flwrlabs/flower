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
"""Utility to validate the `pyproject.toml` file."""


from pathlib import Path
from typing import Any

import click
import tomli

from flwr.cli.typing import SuperLinkConnection
from flwr.common.config import (
    fuse_dicts,
    get_fab_config,
    get_metadata_from_config,
    parse_config_args,
    validate_config,
)


def get_fab_metadata(fab_file: Path | bytes) -> tuple[str, str]:
    """Extract the fab_id and the fab_version from a FAB file or path.

    Parameters
    ----------
    fab_file : Union[Path, bytes]
        The Flower App Bundle file to validate and extract the metadata from.
        It can either be a path to the file or the file itself as bytes.

    Returns
    -------
    Tuple[str, str]
        The `fab_id` and `fab_version` of the given Flower App Bundle.
    """
    return get_metadata_from_config(get_fab_config(fab_file))


def load_and_validate(
    path: Path | None = None,
    check_module: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    """Load and validate pyproject.toml as dict.

    Parameters
    ----------
    path : Optional[Path] (default: None)
        The path of the Flower App config file to load. By default it
        will try to use `pyproject.toml` inside the current directory.
    check_module: bool (default: True)
        Whether the validity of the Python module should be checked.
        This requires the project to be installed in the currently
        running environment. True by default.

    Returns
    -------
    tuple[dict[str, Any], list[str]]
        A tuple of the loaded configuration as dictionary and a list of warnings.

    Raises
    ------
    ValueError
        If the configuration is invalid or the file cannot be loaded.
    """
    if path is None:
        path = Path.cwd() / "pyproject.toml"
    path = path.resolve()
    config = load(path)

    if config is None:
        raise ValueError(
            f"Failed to load Flower App configuration in '{path}'. "
            "File may be missing or invalid."
        )

    is_valid, errors, warnings = validate_config(config, check_module, path.parent)

    if not is_valid:
        raise ValueError(
            f"Invalid Flower App configuration in '{path}':\n"
            + "\n".join([f"- {line}" for line in errors])
        )

    return config, warnings


def load(toml_path: Path) -> dict[str, Any] | None:
    """Load pyproject.toml and return as dict.

    Parameters
    ----------
    toml_path : Path
        Path to the pyproject.toml file.

    Returns
    -------
    dict[str, Any] | None
        Parsed TOML configuration as dictionary, or None if file doesn't exist
        or has invalid TOML syntax.
    """
    if not toml_path.is_file():
        return None

    with toml_path.open("rb") as toml_file:
        try:
            return tomli.load(toml_file)
        except tomli.TOMLDecodeError:
            return None


def validate_federation_in_project_config(
    federation: str | None,
    config: dict[str, Any],
    overrides: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Validate the federation name in the Flower project configuration.

    Parameters
    ----------
    federation : str | None
        Name of the federation, or None to use default from config.
    config : dict[str, Any]
        The project configuration dictionary.
    overrides : list[str] | None
        List of configuration override strings. Default is None.

    Returns
    -------
    tuple[str, dict[str, Any]]
        A tuple of (federation_name, federation_config).

    Raises
    ------
    click.ClickException
        If no federation name provided and no default found, or if federation
        doesn't exist in config.
    """
    federation = federation or config["tool"]["flwr"]["federations"].get("default")

    if federation is None:
        raise click.ClickException(
            "No federation name was provided and the project's `pyproject.toml` "
            "doesn't declare a default federation (with an Control API address or an "
            "`options.num-supernodes` value)."
        )

    # Validate the federation exists in the configuration
    federation_config = config["tool"]["flwr"]["federations"].get(federation)
    if federation_config is None:
        available_feds = {
            fed for fed in config["tool"]["flwr"]["federations"] if fed != "default"
        }
        raise click.ClickException(
            f"There is no `{federation}` federation declared in the "
            "`pyproject.toml`.\n The following federations were found:\n\n"
            + "\n".join(available_feds)
        )

    # Override the federation configuration if provided
    if overrides:
        overrides_dict = parse_config_args(overrides, flatten=False)
        federation_config = fuse_dicts(federation_config, overrides_dict)

    return federation, federation_config


def load_certificate_in_connection(
    connection: SuperLinkConnection,
) -> bytes | None:
    """Validate TLS-related settings and load root certificates if provided.

    Parameters
    ----------
    connection : SuperLinkConnection
        The SuperLink connection configuration.

    Returns
    -------
    bytes | None
        The loaded root certificate bytes if a custom certificate is configured.
        None if TLS is disabled or if gRPC should use its default trust store.

    Raises
    ------
    ValueError
        If required TLS settings are missing.
    click.ClickException
        If the configuration is invalid or the certificate file cannot be read.
    """
    # Process root certificates
    if root_certificates := connection.root_certificates:
        if connection.insecure:
            raise click.ClickException(
                "`root-certificates` were provided but the `insecure` parameter "
                "is set to `True`."
            )

        # TLS is enabled with self-signed certificates: attempt to read the file
        try:
            root_certificates_bytes = Path(root_certificates).read_bytes()
        except Exception as e:
            raise click.ClickException(
                f"Failed to read certificate file `{root_certificates}`: {e}"
            ) from e
    else:
        root_certificates_bytes = None

    return root_certificates_bytes


def get_insecure_flag(federation_config: dict[str, Any]) -> bool:
    """Extract and validate the `insecure` flag from the federation configuration.

    Parameters
    ----------
    federation_config : dict[str, Any]
        The federation configuration dictionary.

    Returns
    -------
    bool
        The insecure flag value. Returns False if not specified.

    Raises
    ------
    click.ClickException
        If insecure value is not a boolean type.
    """
    insecure_value = federation_config.get("insecure")

    if insecure_value is None:
        # Not provided, default to False (TLS enabled)
        return False
    if isinstance(insecure_value, bool):
        return insecure_value
    raise click.ClickException(
        "Invalid type for `insecure`: expected a boolean if provided. "
        "(`insecure = true` or `insecure = false`)"
    )
