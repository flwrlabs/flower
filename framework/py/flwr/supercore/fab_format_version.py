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
"""FAB format version validation and derived metadata helpers."""


from dataclasses import dataclass
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version


@dataclass(frozen=True)
class FabFormatMetadata:
    """Normalized metadata derived during FAB build and returned to callers."""

    fab_format_version: int
    flwr_version_min: str | None
    flwr_version_target: str | None


def _parse_version(value: str, field_name: str) -> Version:
    """Parse a version string and raise a consistent config error if invalid."""
    try:
        return Version(value)
    except InvalidVersion as err:
        raise ValueError(f"Invalid {field_name}: expected a valid version.") from err


def _get_flwr_requirement(config: dict[str, Any]) -> Requirement | None:
    """Return the unique `flwr` dependency declared in `[project].dependencies`."""
    dependencies = config.get("project", {}).get("dependencies")
    if dependencies is None:
        return None

    if not isinstance(dependencies, list):
        raise ValueError(
            'Invalid [project].dependencies: expected a list containing a "flwr" '
            "dependency declaration."
        )

    flwr_requirements: list[Requirement] = []
    for dependency in dependencies:
        if not isinstance(dependency, str):
            raise ValueError(
                "Invalid [project].dependencies: dependency entries must be strings."
            )
        try:
            requirement = Requirement(dependency)
        except InvalidRequirement as err:
            raise ValueError(
                f'Invalid dependency declaration "{dependency}": {err}'
            ) from err

        if requirement.name == "flwr":
            flwr_requirements.append(requirement)

    if not flwr_requirements:
        return None

    if len(flwr_requirements) > 1:
        raise ValueError('Multiple "flwr" dependency declarations are not supported.')

    return flwr_requirements[0]


def _get_declared_license_file(config: dict[str, Any]) -> str | None:
    """Return `[project].license.file` when declared and valid."""
    project_config = config.get("project")
    if not isinstance(project_config, dict):
        return None

    license_entry = project_config.get("license")
    if license_entry is None:
        return None

    if isinstance(license_entry, str):
        raise ValueError(
            "fab-format-version = 1 requires [project].license to be "
            '{ file = "LICENSE" } or { file = "LICENSE.md" }.'
        )

    if not isinstance(license_entry, dict):
        raise ValueError("Invalid [project].license: expected a string or table value.")

    if "text" in license_entry:
        raise ValueError(
            "fab-format-version = 1 requires [project].license to reference "
            "a root-level license file, not inline text."
        )

    if "file" not in license_entry:
        raise ValueError(
            "fab-format-version = 1 requires [project].license to be "
            '{ file = "LICENSE" } or { file = "LICENSE.md" }.'
        )

    license_file = license_entry["file"]
    if not isinstance(license_file, str):
        raise ValueError("Invalid [project].license.file: expected a string path.")

    if license_file not in {"LICENSE", "LICENSE.md"}:
        raise ValueError(
            "fab-format-version = 1 requires [project].license.file to be "
            '"LICENSE" or "LICENSE.md".'
        )

    return license_file


def _require_license_file(config: dict[str, Any]) -> str:
    """Return the required license file declaration for `fab-format-version = 1`."""
    license_file = _get_declared_license_file(config)
    if license_file is None:
        raise ValueError(
            "fab-format-version = 1 requires [project].license to be "
            '{ file = "LICENSE" } or { file = "LICENSE.md" }.'
        )
    return license_file


def _derive_flwr_version_min(requirement: Requirement) -> Version:
    """Derive the supported Flower minimum version from `>=` specifiers only."""
    lower_bound: Version | None = None

    for specifier in requirement.specifier:
        if specifier.operator != ">=":
            # NOTE: Only inclusive lower bounds contribute to derived FAB metadata.
            continue

        version = _parse_version(specifier.version, '"flwr" dependency specifier')
        if lower_bound is None or version > lower_bound:
            lower_bound = version

    if lower_bound is None:
        raise ValueError(
            'Invalid "flwr" dependency specifier: an inclusive lower bound declared '
            'with ">=" is required for fab-format-version = 1.'
        )

    return lower_bound


def _get_flwr_app_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the `[tool.flwr.app]` table without mutating the parsed config."""
    tool_config = config.get("tool")
    if not isinstance(tool_config, dict):
        return {}

    flwr_config = tool_config.get("flwr")
    if not isinstance(flwr_config, dict):
        return {}

    app_config = flwr_config.get("app")
    return app_config if isinstance(app_config, dict) else {}


def _resolve_fab_format_version(app_config: dict[str, Any]) -> int:
    """Resolve and validate `fab-format-version` from app config."""
    fab_format_version = app_config.get("fab-format-version", 0)
    if (
        not isinstance(fab_format_version, int)
        or isinstance(fab_format_version, bool)
        or fab_format_version < 0
    ):
        raise ValueError(
            "Invalid [tool.flwr.app].fab-format-version: expected a non-negative "
            "integer."
        )
    return fab_format_version


def _parse_flwr_target_version(app_config: dict[str, Any]) -> Version | None:
    """Parse optional `flwr-version-target` from app config."""
    target_raw = app_config.get("flwr-version-target")
    if target_raw is not None and not isinstance(target_raw, str):
        raise ValueError(
            "Invalid [tool.flwr.app].flwr-version-target: expected a string."
        )

    if isinstance(target_raw, str):
        return _parse_version(
            target_raw,
            "[tool.flwr.app].flwr-version-target",
        )
    return None


def _require_flwr_target_version(app_config: dict[str, Any]) -> Version:
    """Parse required `flwr-version-target` from app config."""
    target_version = _parse_flwr_target_version(app_config)
    if target_version is None:
        raise ValueError(
            "Missing [tool.flwr.app].flwr-version-target for "
            "fab-format-version >= 1."
        )
    return target_version


def _validate_target_against_requirement(
    target_version: Version | None,
    requirement: Requirement | None,
    lower_bound: Version | None,
) -> None:
    """Ensure `flwr-version-target` satisfies the declared `flwr` requirement."""
    if target_version is None or requirement is None:
        return

    if lower_bound is not None and target_version < lower_bound:
        raise ValueError(
            "Invalid [tool.flwr.app].flwr-version-target: must be greater than or "
            'equal to the declared "flwr" dependency lower bound.'
        )

    if target_version not in requirement.specifier:
        raise ValueError(
            "Invalid [tool.flwr.app].flwr-version-target: must satisfy the "
            'declared "flwr" dependency specifier.'
        )


def _build_fab_metadata(
    fab_format_version: int,
    target_version: Version | None,
    lower_bound: Version | None,
) -> FabFormatMetadata:
    """Create normalized metadata from parsed versions."""
    return FabFormatMetadata(
        fab_format_version=fab_format_version,
        flwr_version_min=str(lower_bound) if lower_bound is not None else None,
        flwr_version_target=str(target_version) if target_version is not None else None,
    )


def _normalize_and_validate_fab_format_v0(config: dict[str, Any]) -> FabFormatMetadata:
    """Derive best-effort compatibility metadata for `fab-format-version = 0`.

    - `flwr` dependency is optional.
    - `flwr-version-target` is ignored.
    - Compatibility minimum is derived from the highest declared `>=` specifier.
    - All non-`>=` specifiers are ignored for metadata derivation.
    """
    requirement = _get_flwr_requirement(config)
    lower_bound: Version | None = None

    if requirement is not None:
        try:
            lower_bound = _derive_flwr_version_min(requirement)
        except ValueError:
            lower_bound = None

    return _build_fab_metadata(0, None, lower_bound)


def _normalize_and_validate_fab_format_v1(config: dict[str, Any]) -> FabFormatMetadata:
    """Require and derive strict metadata for `fab-format-version = 1`.

    - `[project].license` must reference a root-level license file.
    - `flwr` dependency is required.
    - The dependency must include at least one inclusive lower bound declared with
      `>=`.
    - The highest declared `>=` specifier is used as the derived lower bound.
    - All non-`>=` specifiers are ignored for metadata derivation.
    - `flwr-version-target` is required, must be greater than or equal to the
      derived lower bound, and must satisfy the declared `flwr` dependency
      specifier.
    """
    app_config = _get_flwr_app_config(config)
    target_version = _require_flwr_target_version(app_config)
    _require_license_file(config)
    requirement = _get_flwr_requirement(config)
    if requirement is None:
        raise ValueError(
            'Missing "flwr" dependency in [project].dependencies for '
            "fab-format-version >= 1."
        )

    lower_bound = _derive_flwr_version_min(requirement)
    _validate_target_against_requirement(target_version, requirement, lower_bound)
    return _build_fab_metadata(1, target_version, lower_bound)


def _validate_fab_format_v0_contents(
    _config: dict[str, Any], _filtered_paths: list[str]
) -> None:
    """Validate the final FAB contents for `fab-format-version = 0`."""
    # Reserved for future file-level FAB format rules.
    return None


def _validate_fab_format_v1_contents(
    config: dict[str, Any], filtered_paths: list[str]
) -> None:
    """Validate the final FAB contents for `fab-format-version = 1`."""
    license_file = _require_license_file(config)
    if license_file not in filtered_paths:
        raise ValueError(
            "fab-format-version = 1 requires the declared [project].license.file "
            "to be included in the FAB."
        )


def normalize_and_validate_fab_format(
    config: dict[str, Any],
) -> FabFormatMetadata:
    """Normalize FAB metadata in config and validate `fab-format-version` rules."""
    app_config = _get_flwr_app_config(config)
    fab_format_version = _resolve_fab_format_version(app_config)
    if fab_format_version == 0:
        return _normalize_and_validate_fab_format_v0(config)
    if fab_format_version == 1:
        return _normalize_and_validate_fab_format_v1(config)

    raise ValueError(
        f"Unsupported [tool.flwr.app].fab-format-version: {fab_format_version}."
    )


def validate_fab_files_for_format(
    config: dict[str, Any], filtered_paths: list[str]
) -> None:
    """Validate the final FAB contents using the selected format ruleset."""
    app_config = _get_flwr_app_config(config)
    fab_format_version = _resolve_fab_format_version(app_config)
    if fab_format_version == 0:
        _validate_fab_format_v0_contents(config, filtered_paths)
        return
    if fab_format_version == 1:
        _validate_fab_format_v1_contents(config, filtered_paths)
        return

    raise ValueError(
        f"Unsupported [tool.flwr.app].fab-format-version: {fab_format_version}."
    )
