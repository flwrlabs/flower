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

from packaging.version import InvalidVersion, Version

from flwr.supercore.constant import (
    FLWR_COMPONENT_NAME_METADATA_KEY,
    FLWR_PACKAGE_NAME_METADATA_KEY,
    FLWR_PACKAGE_VERSION_METADATA_KEY,
)
from flwr.supercore.utils import get_metadata_str_checked
from flwr.supercore.version import package_name as flwr_package_name
from flwr.supercore.version import package_version as flwr_package_version

_SUPPORTED_FLOWER_PACKAGE_NAMES = frozenset({"flwr", "flwr-nightly"})
_RUNTIME_METADATA_KEYS = (
    FLWR_PACKAGE_NAME_METADATA_KEY,
    FLWR_PACKAGE_VERSION_METADATA_KEY,
    FLWR_COMPONENT_NAME_METADATA_KEY,
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
        values_by_key = {
            key: get_metadata_str_checked(grpc_metadata, key)
            for key in _RUNTIME_METADATA_KEYS
        }

        if all(error == "missing" for _, error in values_by_key.values()):
            return None, None

        for error_kind, message_prefix in (
            ("missing", "Missing required Flower runtime metadata: "),
            ("duplicate", "Flower runtime metadata contains duplicate values: "),
            ("wrong_type", "Flower runtime metadata contains non-string values: "),
            ("empty", "Flower runtime metadata contains empty values: "),
        ):
            matching_keys = [
                key for key, (_, error) in values_by_key.items() if error == error_kind
            ]
            if matching_keys:
                matching_keys_str = ", ".join(sorted(matching_keys))
                return None, f"{message_prefix}{matching_keys_str}."

        package_name = values_by_key[FLWR_PACKAGE_NAME_METADATA_KEY][0]
        package_version = values_by_key[FLWR_PACKAGE_VERSION_METADATA_KEY][0]
        component_name = values_by_key[FLWR_COMPONENT_NAME_METADATA_KEY][0]
        assert package_name is not None
        assert package_version is not None
        assert component_name is not None

        return (
            cls(
                package_name=package_name,
                package_version=package_version,
                component_name=component_name,
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


def get_runtime_version_rejection(
    connection_name: str,
    local_metadata: RuntimeVersionMetadata,
    peer_metadata: RuntimeVersionMetadata | None,
) -> str | None:
    """Return a rejection message, or `None` when the peer should continue."""
    if peer_metadata is None:
        return None

    if local_metadata.package_name.strip() not in _SUPPORTED_FLOWER_PACKAGE_NAMES:
        return None

    package_name_error = _get_package_name_error(local_metadata, peer_metadata)
    if package_name_error is not None:
        return format_invalid_metadata_message(connection_name, package_name_error)

    local_version, local_error = _parse_runtime_version(
        local_metadata.package_version, subject="Local"
    )
    if local_error is not None:
        return format_invalid_metadata_message(connection_name, local_error)

    peer_version, peer_error = _parse_runtime_version(
        peer_metadata.package_version, subject="Peer"
    )
    if peer_error is not None:
        return format_invalid_metadata_message(connection_name, peer_error)

    assert local_version is not None
    assert peer_version is not None

    if (
        local_version.major == peer_version.major
        and local_version.minor == peer_version.minor
    ):
        return None

    return format_incompatible_version_message(
        connection_name,
        local_metadata,
        peer_metadata,
    )


def _get_package_name_error(
    local_metadata: RuntimeVersionMetadata,
    peer_metadata: RuntimeVersionMetadata,
) -> str | None:
    """Return an error when the peer package name is not first-party.

    The local runtime can legitimately report `unknown` in source/non-installed
    environments, so that case must not turn the receiver into a hard-failing
    gate for every incoming RPC.
    """
    local_package_name = local_metadata.package_name.strip()
    if local_package_name not in _SUPPORTED_FLOWER_PACKAGE_NAMES:
        return None

    peer_package_name = peer_metadata.package_name.strip()
    if peer_package_name not in _SUPPORTED_FLOWER_PACKAGE_NAMES:
        return (
            "Peer Flower package name is not recognized: "
            f"{peer_metadata.package_name!r}."
        )

    return None


def _parse_runtime_version(
    package_version: str, *, subject: str
) -> tuple[Version | None, str | None]:
    """Parse a runtime version string or return the invalid-metadata reason."""
    try:
        return Version(package_version), None
    except InvalidVersion:
        return (
            None,
            f"{subject} Flower version metadata cannot be parsed: {package_version!r}.",
        )
