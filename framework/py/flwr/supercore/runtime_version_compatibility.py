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
from flwr.supercore.utils import (
    MetadataLookupError,
    find_metadata_keys,
    get_metadata_str_checked,
)
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

        # Check version validity when the package name is recognized
        if package_name_value != "unknown":
            try:
                Version(package_version_value)
            except InvalidVersion:
                raise ValueError(
                    "`package_version_value` is not a valid version: "
                    f"{package_version_value!r}"
                ) from None

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
        # TEMPORARY: allow continuation when all runtime metadata keys are missing
        # to avoid hard-failing older clients without metadata
        if _metadata_is_missing(grpc_metadata):
            return None, None

        try:
            ret = RuntimeVersionMetadata(
                package_name=get_metadata_str_checked(
                    grpc_metadata, FLWR_PACKAGE_NAME_METADATA_KEY
                ),
                package_version=get_metadata_str_checked(
                    grpc_metadata, FLWR_PACKAGE_VERSION_METADATA_KEY
                ),
                component_name=get_metadata_str_checked(
                    grpc_metadata, FLWR_COMPONENT_NAME_METADATA_KEY
                ),
            )
            return ret, None

        except MetadataLookupError as e:
            return None, f"Invalid Flower runtime metadata: {str(e)}"

    def append_to_grpc_metadata(
        self,
        grpc_metadata: Sequence[tuple[str, str | bytes]] | None,
    ) -> tuple[tuple[str, str | bytes], ...]:
        """Return gRPC metadata with runtime version values added."""
        metadata = tuple(grpc_metadata or ())
        existing_runtime_keys = find_metadata_keys(metadata, _RUNTIME_METADATA_KEYS)
        if existing_runtime_keys:
            raise RuntimeError(
                "gRPC metadata already contains runtime version keys: "
                f"{', '.join(sorted(existing_runtime_keys))}"
            )
        runtime_metadata = (
            (FLWR_PACKAGE_NAME_METADATA_KEY, self.package_name),
            (FLWR_PACKAGE_VERSION_METADATA_KEY, self.package_version),
            (FLWR_COMPONENT_NAME_METADATA_KEY, self.component_name),
        )
        return metadata + runtime_metadata

    def check_compatibility(self, peer: RuntimeVersionMetadata | None) -> str | None:
        """Return a rejection message, or ``None`` if the peer is accepted.

        Rejects the peer if any of the following are true:
        - The peer's Flower package name is not recognized.
        - The peer's Flower version cannot be parsed as a valid version.
        - The peer's major or minor version differs from the local version.

        Accepts the peer (returns ``None``) if any of the following are true:
        - The peer metadata is missing (temporary allowance for older clients).
        - The local package name is not recognized.
        - The peer's major and minor version match the local version.
        """
        # TEMPORARY: allow continuation when peer metadata is missing to avoid
        # hard-failing older clients without metadata
        if peer is None:
            return None

        # Reject suspicious peer package name
        peer_package_name = peer.package_name.strip()
        if peer_package_name not in _SUPPORTED_FLOWER_PACKAGE_NAMES:
            return f"Peer Flower package name is not recognized: {peer_package_name!r}."

        # Allow continuation when the local package name is not recognized
        if self.package_name.strip() not in _SUPPORTED_FLOWER_PACKAGE_NAMES:
            return None

        # Parse versions
        local_version = Version(self.package_version)
        try:
            peer_version = Version(peer.package_version)
        except InvalidVersion:
            return (
                f"Peer Flower version metadata cannot be parsed: "
                f"{peer.package_version!r}."
            )

        # Check major.minor compatibility
        if (
            local_version.major != peer_version.major
            or local_version.minor != peer_version.minor
        ):
            return (
                f"{self.component_name} version {self.package_version} only accepts "
                "peers from the same major.minor release, but received "
                f"{peer.component_name} version {peer.package_version}."
            )

        # Versions are compatible
        return None


def _metadata_is_missing(
    metadata: Sequence[tuple[str, str | bytes]] | None,
) -> bool:
    """Return `True` if all runtime metadata keys are missing from the gRPC metadata.

    This is a TEMPORARY helper to allow older clients without runtime metadata to
    continue working rather than being rejected for missing metadata. It is safe to
    remove this once the minimum supported Flower version is new enough that all clients
    are expected to include runtime metadata.
    """
    if metadata is None:
        return True

    metadata_keys = {key for key, _ in metadata}
    return all(key not in metadata_keys for key in _RUNTIME_METADATA_KEYS)
