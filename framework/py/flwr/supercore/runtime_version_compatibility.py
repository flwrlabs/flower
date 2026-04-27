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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from packaging.version import InvalidVersion, Version

from flwr.supercore.constant import (
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

_NIGHTLY_VERSION_PATTERN = re.compile(
    r"^(?P<release>(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))"
    r"-nightly(?:[.\-+].*)?$"
)


@dataclass(frozen=True)
class RuntimeVersionMetadata:
    """Flower runtime version metadata attached to a caller."""

    package_name: str
    package_version: str
    component_name: str

    @classmethod
    def from_local_component(
        cls,
        component_name: str,
        *,
        package_name_value: str = package_name,
        package_version_value: str = package_version,
    ) -> RuntimeVersionMetadata:
        """Build metadata for the local Flower runtime component."""
        component_name = component_name.strip()
        if component_name == "":
            raise ValueError("`component_name` must be a non-empty string")
        return cls(
            package_name=package_name_value,
            package_version=package_version_value,
            component_name=component_name,
        )

    @classmethod
    def from_grpc_metadata(
        cls,
        grpc_metadata: Sequence[tuple[str, str | bytes]] | None,
    ) -> tuple[RuntimeVersionMetadata | None, str | None]:
        """Parse runtime version metadata from a gRPC metadata sequence."""
        relevant_keys = (
            FLWR_PACKAGE_NAME_METADATA_KEY,
            FLWR_PACKAGE_VERSION_METADATA_KEY,
            FLWR_COMPONENT_NAME_METADATA_KEY,
        )
        metadata_values = _coerce_grpc_metadata_values(
            grpc_metadata, relevant_keys=relevant_keys
        )
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
                "Flower runtime metadata contains duplicate values: "
                f"{duplicate_keys_str}.",
            )

        missing_keys = [key for key in relevant_keys if key not in metadata_values]
        if missing_keys:
            missing_keys_str = ", ".join(sorted(missing_keys))
            return (
                None,
                f"Missing required Flower runtime metadata: {missing_keys_str}.",
            )

        values = {
            # each relevant key appears exactly once
            key: metadata_values[key][0].strip()
            for key in relevant_keys
        }
        empty_keys = [key for key, value in values.items() if value == ""]
        if empty_keys:
            empty_keys_str = ", ".join(sorted(empty_keys))
            return (
                None,
                f"Flower runtime metadata contains empty values: {empty_keys_str}.",
            )

        return (
            cls(
                package_name=values[FLWR_PACKAGE_NAME_METADATA_KEY],
                package_version=values[FLWR_PACKAGE_VERSION_METADATA_KEY],
                component_name=values[FLWR_COMPONENT_NAME_METADATA_KEY],
            ),
            None,
        )

    def to_dict(self) -> dict[str, str]:
        """Serialize runtime version metadata to a string dictionary."""
        return {
            FLWR_PACKAGE_NAME_METADATA_KEY: self.package_name,
            FLWR_PACKAGE_VERSION_METADATA_KEY: self.package_version,
            FLWR_COMPONENT_NAME_METADATA_KEY: self.component_name,
        }

    def append_to_grpc_metadata(
        self,
        grpc_metadata: Sequence[tuple[str, str | bytes]] | None,
    ) -> tuple[tuple[str, str | bytes], ...]:
        """Return gRPC metadata with runtime version values added or replaced."""
        metadata = tuple(grpc_metadata or ())
        runtime_keys = self.to_dict()
        filtered_metadata = tuple(
            (key, value) for key, value in metadata if key not in runtime_keys
        )
        return filtered_metadata + tuple(runtime_keys.items())

    def check_compatibility_with(
        self,
        peer_metadata: RuntimeVersionMetadata | None,
    ) -> CompatibilityResult:
        """Evaluate whether a peer is runtime-compatible with the local component."""
        if peer_metadata is None:
            return CompatibilityResult(
                status="missing",
                reason=None,
                local_metadata=self,
                peer_metadata=None,
                local_version=None,
                peer_version=None,
            )

        local_version = parse_flower_version(self.package_version)
        if local_version is None:
            return CompatibilityResult(
                status="disabled",
                reason=(
                    "Local Flower version metadata cannot be parsed, version checks "
                    f"are disabled: {self.package_version!r}."
                ),
                local_metadata=self,
                peer_metadata=peer_metadata,
                local_version=None,
                peer_version=None,
            )

        peer_version = parse_flower_version(peer_metadata.package_version)
        if peer_version is None:
            peer_version_repr = repr(peer_metadata.package_version)
            return CompatibilityResult(
                status="disabled",
                reason=(
                    "Peer Flower version metadata cannot be parsed, version checks "
                    f"are disabled: {peer_version_repr}."
                ),
                local_metadata=self,
                peer_metadata=peer_metadata,
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
                local_metadata=self,
                peer_metadata=peer_metadata,
                local_version=local_version,
                peer_version=peer_version,
            )

        return CompatibilityResult(
            status="incompatible",
            reason=(
                "Peer Flower version is outside the accepted major.minor release: "
                f"{peer_metadata.package_version!r}."
            ),
            local_metadata=self,
            peer_metadata=peer_metadata,
            local_version=local_version,
            peer_version=peer_version,
        )


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
    return RuntimeVersionMetadata.from_local_component(
        component_name,
        package_name_value=package_name_value,
        package_version_value=package_version_value,
    )


def runtime_version_metadata_to_dict(
    metadata: RuntimeVersionMetadata,
) -> dict[str, str]:
    """Serialize runtime version metadata to a string dictionary."""
    return metadata.to_dict()


def parse_flower_version(version: str) -> ParsedFlowerVersion | None:
    """Parse a Flower version into its leading `major.minor.patch` tuple."""
    normalized_version = _normalize_flower_version(version)
    try:
        parsed_version = Version(normalized_version)
    except InvalidVersion:
        return None

    if len(parsed_version.release) < 3:
        return None

    return ParsedFlowerVersion(
        major=parsed_version.major,
        minor=parsed_version.minor,
        patch=parsed_version.micro,
    )


def read_runtime_version_metadata(
    metadata: Mapping[str, str] | Iterable[tuple[str, str]] | None,
) -> tuple[RuntimeVersionMetadata | None, str | None]:
    """Read Flower runtime metadata from a mapping or metadata item iterable."""
    return RuntimeVersionMetadata.from_grpc_metadata(_coerce_grpc_metadata(metadata))


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

    return local_metadata.check_compatibility_with(peer)


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


def _coerce_grpc_metadata(
    metadata: Mapping[str, str] | Iterable[tuple[str, str]] | None,
) -> tuple[tuple[str, str | bytes], ...] | None:
    if metadata is None:
        return None
    if isinstance(metadata, Mapping):
        return tuple((str(key), str(value)) for key, value in metadata.items())
    return tuple((str(key), str(value)) for key, value in metadata)


def _coerce_grpc_metadata_values(
    metadata: Sequence[tuple[str, str | bytes]] | None,
    *,
    relevant_keys: Sequence[str] | None = None,
) -> dict[str, list[str]]:
    if metadata is None:
        return {}

    relevant_keys_set = set(relevant_keys) if relevant_keys is not None else None
    values: dict[str, list[str]] = {}
    for key, value in metadata:
        str_key = str(key)
        if relevant_keys_set is not None and str_key not in relevant_keys_set:
            continue
        str_value = (
            value.decode("utf-8", errors="strict")
            if isinstance(value, bytes)
            else value
        )
        values.setdefault(str_key, []).append(str_value)
    return values


def _normalize_flower_version(version: str) -> str:
    normalized_version = version.strip()
    if match := _NIGHTLY_VERSION_PATTERN.match(normalized_version):
        return match.group("release")
    return normalized_version
