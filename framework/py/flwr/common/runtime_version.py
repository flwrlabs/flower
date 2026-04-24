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
"""Helpers for Flower runtime version metadata and compatibility checks."""


from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from flwr.common.constant import (
    FLWR_COMPONENT_NAME_METADATA_KEY,
    FLWR_PACKAGE_NAME_METADATA_KEY,
    FLWR_PACKAGE_VERSION_METADATA_KEY,
)
from flwr.supercore.version import package_name, package_version

RuntimeCompatibilityStatus = Literal[
    "missing",
    "disabled",
    "invalid",
    "compatible",
    "incompatible",
]

_VERSION_PATTERN = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
)


@dataclass(frozen=True)
class RuntimeVersionMetadata:
    """Flower runtime version metadata attached to a caller."""

    package_name: str
    package_version: str
    component_name: str


@dataclass(frozen=True, order=True)
class ParsedFlowerVersion:
    """A parsed `major.minor.patch` Flower version."""

    major: int
    minor: int
    patch: int


@dataclass(frozen=True)
class CompatibilityResult:
    """Compatibility decision for a runtime peer."""

    status: RuntimeCompatibilityStatus
    reason: str | None
    local_metadata: RuntimeVersionMetadata
    peer_metadata: RuntimeVersionMetadata | None
    local_version: ParsedFlowerVersion | None
    peer_version: ParsedFlowerVersion | None


def build_runtime_version_metadata(
    component_name: str,
    *,
    package_name_value: str = package_name,
    package_version_value: str = package_version,
) -> RuntimeVersionMetadata:
    """Build metadata for the local Flower runtime component."""
    component_name = component_name.strip()
    if component_name == "":
        raise ValueError("`component_name` must be a non-empty string")
    return RuntimeVersionMetadata(
        package_name=package_name_value,
        package_version=package_version_value,
        component_name=component_name,
    )


def runtime_version_metadata_to_dict(
    metadata: RuntimeVersionMetadata,
) -> dict[str, str]:
    """Serialize runtime version metadata to a string dictionary."""
    return {
        FLWR_PACKAGE_NAME_METADATA_KEY: metadata.package_name,
        FLWR_PACKAGE_VERSION_METADATA_KEY: metadata.package_version,
        FLWR_COMPONENT_NAME_METADATA_KEY: metadata.component_name,
    }


def parse_flower_version(version: str) -> ParsedFlowerVersion | None:
    """Parse a Flower version into its leading `major.minor.patch` tuple."""
    if not (match := _VERSION_PATTERN.match(version.strip())):
        return None
    return ParsedFlowerVersion(
        major=int(match.group("major")),
        minor=int(match.group("minor")),
        patch=int(match.group("patch")),
    )


def read_runtime_version_metadata(
    metadata: Mapping[str, str] | Iterable[tuple[str, str]] | None,
) -> tuple[RuntimeVersionMetadata | None, str | None]:
    """Read Flower runtime metadata from a mapping or metadata item iterable."""
    relevant_keys = (
        FLWR_PACKAGE_NAME_METADATA_KEY,
        FLWR_PACKAGE_VERSION_METADATA_KEY,
        FLWR_COMPONENT_NAME_METADATA_KEY,
    )
    metadata_values = _coerce_metadata_values(metadata)
    present_keys = [key for key in relevant_keys if key in metadata_values]
    if not present_keys:
        return None, None

    duplicate_keys = [
        key for key in relevant_keys if len(metadata_values.get(key, [])) > 1
    ]
    if duplicate_keys:
        duplicate_keys_str = ", ".join(sorted(duplicate_keys))
        return (
            None,
            f"Flower runtime metadata contains duplicate values: {duplicate_keys_str}.",
        )

    metadata_map = {
        key: values[0] for key, values in metadata_values.items() if len(values) == 1
    }

    missing_keys = [key for key in relevant_keys if key not in metadata_map]
    if missing_keys:
        missing_keys_str = ", ".join(sorted(missing_keys))
        return None, f"Missing required Flower runtime metadata: {missing_keys_str}."

    values = {key: metadata_map[key].strip() for key in relevant_keys}
    empty_keys = [key for key, value in values.items() if value == ""]
    if empty_keys:
        empty_keys_str = ", ".join(sorted(empty_keys))
        return None, f"Flower runtime metadata contains empty values: {empty_keys_str}."

    return (
        RuntimeVersionMetadata(
            package_name=values[FLWR_PACKAGE_NAME_METADATA_KEY],
            package_version=values[FLWR_PACKAGE_VERSION_METADATA_KEY],
            component_name=values[FLWR_COMPONENT_NAME_METADATA_KEY],
        ),
        None,
    )


def evaluate_runtime_version_compatibility(
    local_metadata: RuntimeVersionMetadata,
    peer_metadata: (
        RuntimeVersionMetadata | Mapping[str, str] | Iterable[tuple[str, str]] | None
    ),
) -> CompatibilityResult:
    """Evaluate whether a peer is runtime-compatible with the local component."""
    peer: RuntimeVersionMetadata | None
    metadata_error: str | None
    if isinstance(peer_metadata, RuntimeVersionMetadata):
        peer = peer_metadata
        metadata_error = None
    else:
        peer, metadata_error = read_runtime_version_metadata(peer_metadata)

    if metadata_error is not None:
        return CompatibilityResult(
            status="invalid",
            reason=metadata_error,
            local_metadata=local_metadata,
            peer_metadata=peer,
            local_version=None,
            peer_version=None,
        )

    if peer is None:
        return CompatibilityResult(
            status="missing",
            reason=None,
            local_metadata=local_metadata,
            peer_metadata=None,
            local_version=None,
            peer_version=None,
        )

    local_version = parse_flower_version(local_metadata.package_version)
    if local_version is None:
        return CompatibilityResult(
            status="disabled",
            reason=(
                "Local Flower version metadata cannot be parsed, version checks "
                "are disabled: "
                f"{local_metadata.package_version!r}."
            ),
            local_metadata=local_metadata,
            peer_metadata=peer,
            local_version=None,
            peer_version=None,
        )

    peer_version = parse_flower_version(peer.package_version)
    if peer_version is None:
        peer_version_repr = repr(peer.package_version)
        return CompatibilityResult(
            status="disabled",
            reason=(
                "Peer Flower version metadata cannot be parsed, version checks "
                f"are disabled: {peer_version_repr}."
            ),
            local_metadata=local_metadata,
            peer_metadata=peer,
            local_version=local_version,
            peer_version=None,
        )

    if (
        local_version.major == peer_version.major
        and local_version.minor == peer_version.minor
    ):
        return CompatibilityResult(
            status="compatible",
            reason=None,
            local_metadata=local_metadata,
            peer_metadata=peer,
            local_version=local_version,
            peer_version=peer_version,
        )

    return CompatibilityResult(
        status="incompatible",
        reason=(
            "Peer Flower version is outside the accepted major.minor release: "
            f"{peer.package_version!r}."
        ),
        local_metadata=local_metadata,
        peer_metadata=peer,
        local_version=local_version,
        peer_version=peer_version,
    )


def format_incompatible_version_message(
    connection_name: str,
    local_metadata: RuntimeVersionMetadata,
    peer_metadata: RuntimeVersionMetadata,
) -> str:
    """Format the standard incompatible-version error message."""
    return (
        f"Incompatible Flower version for {connection_name}.\n"
        f"Local {local_metadata.component_name} version "
        f"{local_metadata.package_version} only accepts peers from the same "
        f"major.minor release, but received {peer_metadata.component_name} "
        f"version {peer_metadata.package_version}."
    )


def format_invalid_metadata_message(connection_name: str, detail: str) -> str:
    """Format a standard invalid-metadata error message."""
    return f"Invalid Flower version metadata for {connection_name}. {detail}"


def _coerce_metadata_values(
    metadata: Mapping[str, str] | Iterable[tuple[str, str]] | None,
) -> dict[str, list[str]]:
    if metadata is None:
        return {}
    if isinstance(metadata, Mapping):
        return {str(key): [str(value)] for key, value in metadata.items()}

    values: dict[str, list[str]] = {}
    for key, value in metadata:
        str_key = str(key)
        values.setdefault(str_key, []).append(str(value))
    return values
