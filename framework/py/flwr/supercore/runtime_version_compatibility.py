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
from flwr.supercore.utils import get_metadata_str_checked
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
        values_by_key = {
            key: get_metadata_str_checked(grpc_metadata, key) for key in relevant_keys
        }
        present_keys = [
            key for key, result in values_by_key.items() if result.error != "missing"
        ]
        if not present_keys:
            return None, None

        duplicate_keys = [
            key for key, result in values_by_key.items() if result.error == "duplicate"
        ]
        if duplicate_keys:
            duplicate_keys_str = ", ".join(sorted(duplicate_keys))
            return (
                None,
                "Flower runtime metadata contains duplicate values: "
                f"{duplicate_keys_str}.",
            )

        missing_keys = [
            key for key, result in values_by_key.items() if result.error == "missing"
        ]
        if missing_keys:
            missing_keys_str = ", ".join(sorted(missing_keys))
            return (
                None,
                f"Missing required Flower runtime metadata: {missing_keys_str}.",
            )

        wrong_type_keys = [
            key for key, result in values_by_key.items() if result.error == "wrong_type"
        ]
        if wrong_type_keys:
            wrong_type_keys_str = ", ".join(sorted(wrong_type_keys))
            return (
                None,
                "Flower runtime metadata contains non-string values: "
                f"{wrong_type_keys_str}.",
            )

        empty_keys = [
            key for key, result in values_by_key.items() if result.error == "empty"
        ]
        if empty_keys:
            empty_keys_str = ", ".join(sorted(empty_keys))
            return (
                None,
                f"Flower runtime metadata contains empty values: {empty_keys_str}.",
            )

        values: dict[str, str] = {}
        for key in relevant_keys:
            value = values_by_key[key].value
            assert value is not None
            values[key] = value.strip()

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
