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
"""Provide functions for managing global Flower config."""


import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import IO, Any, TypeVar, cast, get_args

import click
import tomli

from flwr.app.user_config import UserConfig, UserConfigValue
from flwr.common.constant import (
    APP_DIR,
    FAB_CONFIG_FILE,
    FAB_EXCLUDE_KEY,
    FAB_HASH_TRUNCATION,
    FAB_INCLUDE_KEY,
)
from flwr.common.typing import Run
from flwr.supercore.fab_format_version import normalize_and_validate_fab_format
from flwr.supercore.utils import get_flwr_home

from . import ConfigRecord, object_ref

T_dict = TypeVar("T_dict", bound=dict[str, Any])  # pylint: disable=invalid-name


def get_project_dir(
    fab_id: str,
    fab_version: str,
    fab_hash: str,
) -> Path:
    """Return the project directory based on the given fab_id and fab_version."""
    # Check the fab_id
    if fab_id.count("/") != 1:
        raise ValueError(
            f"Invalid FAB ID: {fab_id}",
        )
    publisher, project_name = fab_id.split("/")
    return (
        get_flwr_home()
        / APP_DIR
        / f"{publisher}.{project_name}.{fab_version}.{fab_hash[:FAB_HASH_TRUNCATION]}"
    )


def get_project_config(project_dir: str | Path) -> dict[str, Any]:
    """Return pyproject.toml in the given project directory."""
    # Load pyproject.toml file
    toml_path = Path(project_dir) / FAB_CONFIG_FILE
    if not toml_path.is_file():
        raise FileNotFoundError(
            f"Cannot find {FAB_CONFIG_FILE} in {project_dir}",
        )
    with toml_path.open(encoding="utf-8") as toml_file:
        config = tomli.loads(toml_file.read())

    # Validate pyproject.toml fields
    is_valid, errors, _ = validate_fields_in_config(config)
    if not is_valid:
        error_msg = "\n".join([f"  - {error}" for error in errors])
        raise ValueError(
            f"Invalid {FAB_CONFIG_FILE}:\n{error_msg}",
        )

    return config


def fuse_dicts(
    main_dict: T_dict,
    override_dict: T_dict,
    check_keys: bool = True,
) -> T_dict:
    """Merge a config with the overrides.

    If `check_keys` is set to True, an error will be raised if the override
    dictionary contains keys that are not present in the main dictionary.
    Otherwise, only the keys present in the main dictionary will be updated.
    """
    if not isinstance(main_dict, dict) or not isinstance(override_dict, dict):
        raise ValueError("Both dictionaries must be of type dict")

    fused_dict = cast(T_dict, main_dict.copy())

    for key, value in override_dict.items():
        if key in main_dict:
            if isinstance(value, dict):
                fused_dict[key] = fuse_dicts(main_dict[key], value)
            fused_dict[key] = value
        elif check_keys:
            raise ValueError(f"Key '{key}' is not present in the main dictionary")

    return fused_dict


def get_fused_config_from_dir(
    project_dir: Path, override_config: UserConfig
) -> UserConfig:
    """Merge the overrides from a given dict with the config from a Flower App."""
    default_config = get_project_config(project_dir)["tool"]["flwr"]["app"].get(
        "config", {}
    )
    flat_default_config = flatten_dict(default_config)

    return fuse_dicts(flat_default_config, override_config)


def get_fused_config_from_fab(fab_file: Path | bytes, run: Run) -> UserConfig:
    """Fuse default config in a `FAB` with overrides in a `Run`.

    This enables obtaining a run-config without having to install the FAB. This
    function mirrors `get_fused_config_from_dir`. This is useful when the execution
    of the FAB is delegated to a different process.
    """
    default_config = get_fab_config(fab_file)["tool"]["flwr"]["app"].get("config", {})
    flat_config_flat = flatten_dict(default_config)
    return fuse_dicts(flat_config_flat, run.override_config)


def get_fused_config(run: Run) -> UserConfig:
    """Merge the overrides from a `Run` with the config from a FAB.

    Get the config using the fab_id and the fab_version, remove the nesting by adding
    the nested keys as prefixes separated by dots, and fuse it with the override dict.
    """
    # Return empty dict if fab_id or fab_version is empty
    if not run.fab_id or not run.fab_version:
        return {}

    project_dir = get_project_dir(run.fab_id, run.fab_version, run.fab_hash)

    # Return empty dict if project directory does not exist
    if not project_dir.is_dir():
        return {}

    return get_fused_config_from_dir(project_dir, run.override_config)


def flatten_dict(raw_dict: dict[str, Any] | None, parent_key: str = "") -> UserConfig:
    """Flatten dict by joining nested keys with a given separator."""
    if raw_dict is None:
        return {}

    items: list[tuple[str, UserConfigValue]] = []
    separator: str = "."
    for k, v in raw_dict.items():
        new_key = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, parent_key=new_key).items())
        elif isinstance(v, get_args(UserConfigValue)):
            items.append((new_key, cast(UserConfigValue, v)))
        else:
            raise ValueError(
                f"The value for key {k} needs to be of type `int`, `float`, "
                "`bool, `str`, or  a `dict` of those.",
            )
    return dict(items)


def unflatten_dict(flat_dict: dict[str, Any]) -> dict[str, Any]:
    """Unflatten a dict with keys containing separators into a nested dict."""
    unflattened_dict: dict[str, Any] = {}
    separator: str = "."

    for key, value in flat_dict.items():
        parts = key.split(separator)
        d = unflattened_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value

    return unflattened_dict


def parse_config_args(config: list[str] | None, flatten: bool = True) -> dict[str, Any]:
    """Parse separator separated list of key-value pairs separated by '='."""
    overrides: UserConfig = {}

    if config is None:
        return overrides

    # Handle if .toml file is passed
    if len(config) == 1 and config[0].endswith(".toml"):
        with Path(config[0]).expanduser().open("rb") as config_file:
            overrides = flatten_dict(tomli.load(config_file))
        return overrides

    # Regular expression to capture key-value pairs with possible quoted values
    pattern = re.compile(r"(\S+?)=(\'[^\']*\'|\"[^\"]*\"|\S+)")

    flat_overrides = {}
    for config_line in config:
        if config_line:
            # .toml files aren't allowed alongside other configs
            if config_line.endswith(".toml"):
                raise ValueError(
                    "TOML files cannot be passed alongside key-value pairs."
                )

            matches = pattern.findall(config_line)
            toml_str = "\n".join(f"{k} = {v}" for k, v in matches)
            try:
                overrides.update(tomli.loads(toml_str))
                flat_overrides = flatten_dict(overrides) if flatten else overrides
            except tomli.TOMLDecodeError as err:
                raise click.ClickException(
                    "The provided configuration string is in an invalid format. "
                    "The correct format should be, e.g., 'key1=123 key2=false "
                    'key3="string"\', where values must be of type bool, int, '
                    "string, or float. Ensure proper formatting with "
                    "space-separated key-value pairs."
                ) from err

    return flat_overrides


def get_metadata_from_config(config: dict[str, Any]) -> tuple[str, str]:
    """Extract `fab_id` and `fab_version` from a project config."""
    return (
        f"{config['tool']['flwr']['app']['publisher']}/{config['project']['name']}",
        config["project"]["version"],
    )


def user_config_to_configrecord(config: UserConfig) -> ConfigRecord:
    """Construct a `ConfigRecord` out of a `UserConfig`."""
    c_record = ConfigRecord()
    for k, v in config.items():
        c_record[k] = v

    return c_record


def get_fab_config(fab_file: Path | bytes) -> dict[str, Any]:
    """Extract the config from a FAB file or path.

    Parameters
    ----------
    fab_file : Union[Path, bytes]
        The Flower App Bundle file to validate and extract the metadata from.
        It can either be a path to the file or the file itself as bytes.

    Returns
    -------
    Dict[str, Any]
        The `config` of the given Flower App Bundle.
    """
    fab_file_archive: Path | IO[bytes]
    if isinstance(fab_file, bytes):
        fab_file_archive = BytesIO(fab_file)
    elif isinstance(fab_file, Path):
        fab_file_archive = fab_file
    else:
        raise ValueError("fab_file must be either a Path or bytes")

    with zipfile.ZipFile(fab_file_archive, "r") as zipf:
        with zipf.open("pyproject.toml") as file:
            toml_content = file.read().decode("utf-8")
        try:
            conf = tomli.loads(toml_content)
        except tomli.TOMLDecodeError:
            raise ValueError("Invalid TOML content in pyproject.toml") from None

        is_valid, errors, _ = validate_config(conf, check_module=False)
        if not is_valid:
            raise ValueError(errors)

        return conf


def _validate_run_config(config_dict: dict[str, Any], errors: list[str]) -> None:
    for key, value in config_dict.items():
        if isinstance(value, dict):
            _validate_run_config(config_dict[key], errors)
        elif not isinstance(value, get_args(UserConfigValue)):
            raise ValueError(
                f"The value for key {key} needs to be of type `int`, `float`, "
                "`bool, `str`, or  a `dict` of those.",
            )


def check_pattern_list_value(value: Any, key: str) -> str | None:
    """Validate a pattern list value and return an error message or ``None``."""
    if not isinstance(value, list):
        return f'Property "{key}" in [tool.flwr.app] must be a list of strings.'
    if not value:
        return (
            f'Property "{key}" in [tool.flwr.app] must not be an empty list. '
            "Remove the key to use built-in defaults, or specify at least one pattern."
        )
    if any(not isinstance(pattern, str) or pattern.strip() == "" for pattern in value):
        return (
            f'Property "{key}" in [tool.flwr.app] must be a list of non-empty '
            "strings."
        )
    return None


def _validate_pattern_list(
    app_config: dict[str, Any], key: str, errors: list[str]
) -> None:
    """Validate optional list-of-string pattern fields in [tool.flwr.app]."""
    # FAB include/exclude keys are optional, so if not present, we can skip validation
    if key not in app_config:
        return

    error = check_pattern_list_value(app_config[key], key)
    if error:
        errors.append(error)


# pylint: disable=too-many-branches
def validate_fields_in_config(
    config: dict[str, Any],
) -> tuple[bool, list[str], list[str]]:
    """Validate pyproject.toml fields."""
    errors = []
    warnings = []

    if "project" not in config:
        errors.append("Missing [project] section")
    else:
        if "name" not in config["project"]:
            errors.append('Property "name" missing in [project]')
        if "version" not in config["project"]:
            errors.append('Property "version" missing in [project]')
        if "description" not in config["project"]:
            warnings.append('Recommended property "description" missing in [project]')
        if "license" not in config["project"]:
            warnings.append('Recommended property "license" missing in [project]')

    if (
        "tool" not in config
        or "flwr" not in config["tool"]
        or "app" not in config["tool"]["flwr"]
    ):
        errors.append("Missing [tool.flwr.app] section")
    else:
        if "publisher" not in config["tool"]["flwr"]["app"]:
            errors.append('Property "publisher" missing in [tool.flwr.app]')
        _validate_pattern_list(config["tool"]["flwr"]["app"], FAB_INCLUDE_KEY, errors)
        _validate_pattern_list(config["tool"]["flwr"]["app"], FAB_EXCLUDE_KEY, errors)
        if "config" in config["tool"]["flwr"]["app"]:
            _validate_run_config(config["tool"]["flwr"]["app"]["config"], errors)
        if "components" not in config["tool"]["flwr"]["app"]:
            errors.append("Missing [tool.flwr.app.components] section")
        else:
            if "serverapp" not in config["tool"]["flwr"]["app"]["components"]:
                errors.append(
                    'Property "serverapp" missing in [tool.flwr.app.components]'
                )
            if "clientapp" not in config["tool"]["flwr"]["app"]["components"]:
                errors.append(
                    'Property "clientapp" missing in [tool.flwr.app.components]'
                )

    return len(errors) == 0, errors, warnings


def validate_config(
    config: dict[str, Any],
    check_module: bool = True,
    project_dir: str | Path | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Validate pyproject.toml."""
    is_valid, errors, warnings = validate_fields_in_config(config)

    if not is_valid:
        return False, errors, warnings

    try:
        normalize_and_validate_fab_format(config)
    except ValueError as err:
        return False, [str(err)], warnings

    # Validate serverapp
    serverapp_ref = config["tool"]["flwr"]["app"]["components"]["serverapp"]
    is_valid, reason = object_ref.validate(serverapp_ref, check_module, project_dir)

    if not is_valid and isinstance(reason, str):
        return False, [reason], warnings

    # Validate clientapp
    clientapp_ref = config["tool"]["flwr"]["app"]["components"]["clientapp"]
    is_valid, reason = object_ref.validate(clientapp_ref, check_module, project_dir)

    if not is_valid and isinstance(reason, str):
        return False, [reason], warnings

    return True, [], warnings
