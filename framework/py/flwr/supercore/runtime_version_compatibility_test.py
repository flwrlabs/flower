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
"""Tests for Flower runtime version metadata helpers."""


import pytest

from flwr.supercore.constant import (
    FLWR_COMPONENT_NAME_METADATA_KEY,
    FLWR_PACKAGE_NAME_METADATA_KEY,
    FLWR_PACKAGE_VERSION_METADATA_KEY,
)

from .runtime_version_compatibility import RuntimeVersionMetadata


def test_runtime_version_metadata_appends_new_metadata() -> None:
    """Runtime metadata should append the shared key names."""
    metadata = RuntimeVersionMetadata.from_local_component(
        "supernode",
        package_name_value="flwr",
        package_version_value="1.29.0",
    )

    assert metadata.append_to_grpc_metadata(None) == (
        (FLWR_PACKAGE_NAME_METADATA_KEY, "flwr"),
        (FLWR_PACKAGE_VERSION_METADATA_KEY, "1.29.0"),
        (FLWR_COMPONENT_NAME_METADATA_KEY, "supernode"),
    )


def test_runtime_version_metadata_appends_to_grpc_metadata() -> None:
    """Runtime metadata should preserve unrelated metadata when appending."""
    metadata = RuntimeVersionMetadata.from_local_component(
        "simulation",
        package_name_value="flwr",
        package_version_value="1.29.0",
    )

    grpc_metadata = metadata.append_to_grpc_metadata((("x-test", "value"),))

    assert grpc_metadata == (
        ("x-test", "value"),
        (FLWR_PACKAGE_NAME_METADATA_KEY, "flwr"),
        (FLWR_PACKAGE_VERSION_METADATA_KEY, "1.29.0"),
        (FLWR_COMPONENT_NAME_METADATA_KEY, "simulation"),
    )


def test_runtime_version_metadata_append_rejects_preexisting_runtime_keys() -> None:
    """Appending should fail fast when runtime-version keys already exist."""
    metadata = RuntimeVersionMetadata.from_local_component(
        "simulation",
        package_name_value="flwr",
        package_version_value="1.29.0",
    )

    with pytest.raises(
        RuntimeError,
        match="gRPC metadata already contains runtime version keys: "
        "flwr-package-name",
    ):
        metadata.append_to_grpc_metadata(
            (
                (FLWR_PACKAGE_NAME_METADATA_KEY, "old"),
                ("x-test", "value"),
            )
        )


def test_build_runtime_version_metadata_rejects_empty_component_name() -> None:
    """Component names must not be empty."""
    with pytest.raises(ValueError, match="component_name"):
        RuntimeVersionMetadata.from_local_component("")


def test_runtime_version_metadata_from_grpc_returns_missing_for_absent_keys() -> None:
    """Absent Flower metadata should be treated as the rollout missing case."""
    metadata, error = RuntimeVersionMetadata.from_grpc_metadata(
        (("other-header", "value"),)
    )

    assert metadata is None
    assert error is None


def test_runtime_version_metadata_from_grpc_accepts_metadata_item_iterables() -> None:
    """GRPC metadata-style iterables should be supported directly."""
    metadata, error = RuntimeVersionMetadata.from_grpc_metadata(
        (
            (FLWR_PACKAGE_NAME_METADATA_KEY, "flwr"),
            (FLWR_PACKAGE_VERSION_METADATA_KEY, "1.29.0"),
            (FLWR_COMPONENT_NAME_METADATA_KEY, "cli"),
        )
    )

    assert error is None
    assert metadata == RuntimeVersionMetadata(
        package_name="flwr",
        package_version="1.29.0",
        component_name="cli",
    )


@pytest.mark.parametrize(
    ("grpc_metadata", "expected_error"),
    [
        (
            ((FLWR_PACKAGE_NAME_METADATA_KEY, "flwr"),),
            "Invalid Flower runtime metadata: "
            "Metadata key 'flwr-package-version' is missing.",
        ),
        (
            (
                (FLWR_PACKAGE_NAME_METADATA_KEY, b"flwr"),
                (FLWR_PACKAGE_VERSION_METADATA_KEY, b"1.29.0"),
                (FLWR_COMPONENT_NAME_METADATA_KEY, b"cli"),
            ),
            "Invalid Flower runtime metadata: "
            "Metadata key 'flwr-package-name' has a value of the wrong type.",
        ),
        (
            (
                (FLWR_PACKAGE_NAME_METADATA_KEY, "flwr"),
                (FLWR_PACKAGE_VERSION_METADATA_KEY, b"\xff\xfe"),
                (FLWR_COMPONENT_NAME_METADATA_KEY, "cli"),
            ),
            "Invalid Flower runtime metadata: "
            "Metadata key 'flwr-package-version' has a value of the wrong type.",
        ),
        (
            (
                (FLWR_PACKAGE_NAME_METADATA_KEY, "flwr"),
                (FLWR_PACKAGE_VERSION_METADATA_KEY, "1.29.0"),
                (FLWR_PACKAGE_VERSION_METADATA_KEY, "1.29.1"),
                (FLWR_COMPONENT_NAME_METADATA_KEY, "cli"),
            ),
            "Invalid Flower runtime metadata: "
            "Metadata key 'flwr-package-version' has duplicate values.",
        ),
    ],
)
def test_runtime_version_metadata_from_grpc_rejects_invalid_metadata(
    grpc_metadata: tuple[tuple[str, str | bytes], ...],
    expected_error: str,
) -> None:
    """Malformed runtime metadata should be rejected explicitly."""
    metadata, error = RuntimeVersionMetadata.from_grpc_metadata(grpc_metadata)

    assert metadata is None
    assert error == expected_error


@pytest.mark.parametrize(
    ("local", "peer"),
    [
        (
            RuntimeVersionMetadata("flwr", "1.29.0", "superlink"),
            RuntimeVersionMetadata("flwr", "1.29.7", "supernode"),
        ),
        (
            RuntimeVersionMetadata("flwr", "1.30.0.dev20260425", "superlink"),
            RuntimeVersionMetadata("flwr", "1.30.0rc1", "supernode"),
        ),
        (
            RuntimeVersionMetadata("flwr", "1.30.0", "superlink"),
            RuntimeVersionMetadata("flwr-nightly", "1.30.1.dev20260425", "supernode"),
        ),
        (
            RuntimeVersionMetadata("flwr", "1.29.0", "superlink"),
            None,
        ),
    ],
)
def test_runtime_version_metadata_allows_expected_cases(
    local: RuntimeVersionMetadata,
    peer: RuntimeVersionMetadata | None,
) -> None:
    """Compatible peers and absent metadata should continue."""
    assert local.check_compatibility(peer) is None


@pytest.mark.parametrize(
    ("local", "peer", "expected_rejection"),
    [
        (
            RuntimeVersionMetadata("flwr", "1.29.2", "SuperLink"),
            RuntimeVersionMetadata("flwr", "1.30.0", "SuperNode"),
            "SuperLink version 1.29.2 only accepts peers from the same "
            "major.minor release, but received SuperNode version 1.30.0.",
        ),
        (
            RuntimeVersionMetadata("unknown", "unknown", "SuperLink"),
            RuntimeVersionMetadata("flwr", "1.29.0", "flwr-simulation"),
            None,
        ),
        (
            RuntimeVersionMetadata("flwr", "1.29.0", "SuperLink"),
            RuntimeVersionMetadata("flwr", "main", "SuperNode"),
            "Peer Flower version metadata cannot be parsed: 'main'.",
        ),
        (
            RuntimeVersionMetadata("flwr", "1.29.0", "SuperLink"),
            RuntimeVersionMetadata("forked-flower", "1.29.1", "SuperNode"),
            "Peer Flower package name is not recognized: 'forked-flower'.",
        ),
    ],
)
def test_runtime_version_metadata_rejects_expected_cases(
    local: RuntimeVersionMetadata,
    peer: RuntimeVersionMetadata,
    expected_rejection: str | None,
) -> None:
    """Explicitly invalid or incompatible peers should be rejected."""
    assert local.check_compatibility(peer) == expected_rejection
