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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from packaging.version import InvalidVersion, Version

from flwr.supercore.constant import (
    FLWR_COMPONENT_NAME_METADATA_KEY,
    FLWR_PACKAGE_NAME_METADATA_KEY,
    FLWR_PACKAGE_VERSION_METADATA_KEY,
)
from flwr.supercore.version import package_name as flwr_package_name
from flwr.supercore.version import package_version as flwr_package_version

RuntimeCompatibilityStatus = Literal[
    "missing",
    "disabled",
    "invalid",
    "compatible",
    "incompatible",
]


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
        package_name_value: str = flwr_package_name,
        package_version_value: str = flwr_package_version,
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
        metadata_values, metadata_error = _collect_runtime_metadata_values(
            grpc_metadata, relevant_keys=relevant_keys
        )
        if metadata_error is not None:
            return None, metadata_error
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

    def append_to_grpc_metadata(
        self,
        grpc_metadata: Sequence[tuple[str, str | bytes]] | None,
    ) -> tuple[tuple[str, str | bytes], ...]:
        """Return gRPC metadata with runtime version values added or replaced."""
        metadata = tuple(grpc_metadata or ())
        runtime_metadata = (
            (FLWR_PACKAGE_NAME_METADATA_KEY, self.package_name),
            (FLWR_PACKAGE_VERSION_METADATA_KEY, self.package_version),
            (FLWR_COMPONENT_NAME_METADATA_KEY, self.component_name),
        )
        runtime_keys = {key for key, _ in runtime_metadata}
        filtered_metadata = tuple(
            (key, value) for key, value in metadata if key not in runtime_keys
        )
        return filtered_metadata + runtime_metadata

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

        try:
            local_version = Version(self.package_version)
        except InvalidVersion:
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

        try:
            peer_version = Version(peer_metadata.package_version)
        except InvalidVersion:
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


@dataclass(frozen=True)
class CompatibilityResult:
    """Compatibility decision for a runtime peer."""

    status: RuntimeCompatibilityStatus
    reason: str | None
    local_metadata: RuntimeVersionMetadata
    peer_metadata: RuntimeVersionMetadata | None
    local_version: Version | None
    peer_version: Version | None


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


def _collect_runtime_metadata_values(
    metadata: Sequence[tuple[str, str | bytes]] | None,
    *,
    relevant_keys: Sequence[str] | None = None,
) -> tuple[dict[str, list[str]], str | None]:
    """Collect relevant runtime metadata values from a gRPC metadata sequence.

    NOTE: Only the requested runtime-version keys are inspected. Unrelated gRPC
    metadata is ignored so this parser does not accidentally widen the runtime
    version contract.
    """
    if metadata is None:
        return {}, None

    relevant_keys_lookup = set(relevant_keys) if relevant_keys is not None else None
    values: dict[str, list[str]] = {}
    for key, value in metadata:
        if relevant_keys_lookup is not None and key not in relevant_keys_lookup:
            continue
        if not isinstance(value, str):
            return (
                {},
                f"Flower runtime metadata contains non-string values: {key}.",
            )
        values.setdefault(key, []).append(value)
    return values, None
