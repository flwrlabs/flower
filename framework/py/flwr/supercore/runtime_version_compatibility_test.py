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
from packaging.version import Version

from flwr.supercore.constant import (
    FLWR_COMPONENT_NAME_METADATA_KEY,
    FLWR_PACKAGE_NAME_METADATA_KEY,
    FLWR_PACKAGE_VERSION_METADATA_KEY,
)

from .runtime_version_compatibility import (
    RuntimeVersionMetadata,
    format_incompatible_version_message,
    format_invalid_metadata_message,
)


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
    """Runtime metadata should replace stale values and preserve unrelated ones."""
    metadata = RuntimeVersionMetadata.from_local_component(
        "simulation",
        package_name_value="flwr",
        package_version_value="1.29.0",
    )

    grpc_metadata = metadata.append_to_grpc_metadata(
        (
            (FLWR_PACKAGE_NAME_METADATA_KEY, "old"),
            ("x-test", "value"),
        )
    )

    assert grpc_metadata == (
        ("x-test", "value"),
        (FLWR_PACKAGE_NAME_METADATA_KEY, "flwr"),
        (FLWR_PACKAGE_VERSION_METADATA_KEY, "1.29.0"),
        (FLWR_COMPONENT_NAME_METADATA_KEY, "simulation"),
    )


def test_build_runtime_version_metadata_rejects_empty_component_name() -> None:
    """Component names must not be empty."""
    with pytest.raises(ValueError, match="component_name"):
        RuntimeVersionMetadata.from_local_component("")


def test_read_runtime_version_metadata_returns_missing_for_absent_keys() -> None:
    """Absent Flower metadata should be treated as the rollout missing case."""
    metadata, error = RuntimeVersionMetadata.from_grpc_metadata(
        (("other-header", "value"),)
    )

    assert metadata is None
    assert error is None


def test_read_runtime_version_metadata_rejects_partial_metadata() -> None:
    """Partial Flower metadata should be rejected as invalid."""
    metadata, error = RuntimeVersionMetadata.from_grpc_metadata(
        ((FLWR_PACKAGE_NAME_METADATA_KEY, "flwr"),)
    )

    assert metadata is None
    assert error is not None
    assert "Missing required Flower runtime metadata" in error


def test_read_runtime_version_metadata_accepts_metadata_item_iterables() -> None:
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


def test_runtime_version_metadata_rejects_bytes_values() -> None:
    """Runtime metadata keys should reject non-string gRPC values."""
    metadata, error = RuntimeVersionMetadata.from_grpc_metadata(
        (
            (FLWR_PACKAGE_NAME_METADATA_KEY, b"flwr"),
            (FLWR_PACKAGE_VERSION_METADATA_KEY, b"1.29.0"),
            (FLWR_COMPONENT_NAME_METADATA_KEY, b"cli"),
        )
    )

    assert metadata is None
    assert (
        error
        == "Flower runtime metadata contains non-string values: flwr-package-name."
    )


def test_runtime_version_metadata_rejects_non_string_runtime_values() -> None:
    """Relevant runtime metadata keys should reject non-string values."""
    metadata, error = RuntimeVersionMetadata.from_grpc_metadata(
        (
            (FLWR_PACKAGE_NAME_METADATA_KEY, "flwr"),
            (FLWR_PACKAGE_VERSION_METADATA_KEY, b"\xff\xfe"),
            (FLWR_COMPONENT_NAME_METADATA_KEY, "cli"),
        )
    )

    assert metadata is None
    assert (
        error == "Flower runtime metadata contains non-string values: "
        "flwr-package-version."
    )


def test_runtime_version_metadata_ignores_unrelated_binary_headers() -> None:
    """Unrelated binary metadata should not affect runtime metadata parsing."""
    metadata, error = RuntimeVersionMetadata.from_grpc_metadata(
        (
            ("grpc-trace-bin", b"\xff\x00\xfe"),
            ("other-header", "value"),
        )
    )

    assert metadata is None
    assert error is None


def test_read_runtime_version_metadata_rejects_duplicate_values() -> None:
    """Runtime version metadata keys should appear at most once."""
    metadata, error = RuntimeVersionMetadata.from_grpc_metadata(
        (
            (FLWR_PACKAGE_NAME_METADATA_KEY, "flwr"),
            (FLWR_PACKAGE_VERSION_METADATA_KEY, "1.29.0"),
            (FLWR_PACKAGE_VERSION_METADATA_KEY, "1.29.1"),
            (FLWR_COMPONENT_NAME_METADATA_KEY, "cli"),
        )
    )

    assert metadata is None
    assert (
        error
        == "Flower runtime metadata contains duplicate values: flwr-package-version."
    )


def test_runtime_version_metadata_accepts_same_major_minor() -> None:
    """Patch differences are compatible."""
    local = RuntimeVersionMetadata("flwr", "1.29.0", "superlink")

    result = local.check_compatibility_with(
        RuntimeVersionMetadata("flwr", "1.29.7", "supernode")
    )

    assert result.status == "compatible"
    assert result.peer_version == Version("1.29.7")


def test_runtime_version_metadata_accepts_dev_versions() -> None:
    """PEP 440 nightly/dev versions should be compatible by release tuple."""
    local = RuntimeVersionMetadata("flwr", "1.30.0.dev20260425", "superlink")

    result = local.check_compatibility_with(
        RuntimeVersionMetadata("flwr", "1.30.0rc1", "supernode")
    )

    assert result.status == "compatible"
    assert result.local_version == Version("1.30.0.dev20260425")
    assert result.peer_version == Version("1.30.0rc1")


def test_runtime_version_metadata_rejects_different_minor() -> None:
    """Different minor versions should be incompatible."""
    peer = RuntimeVersionMetadata("flwr", "1.30.0", "supernode")
    local = RuntimeVersionMetadata("flwr", "1.29.2", "superlink")

    result = local.check_compatibility_with(peer)

    assert result.status == "incompatible"
    assert result.peer_metadata == peer
    assert (
        format_incompatible_version_message(
            "SuperNode <-> SuperLink Fleet API", local, peer
        )
        == "Incompatible Flower version for SuperNode <-> SuperLink Fleet API.\n"
        "Local superlink version 1.29.2 only accepts peers from the same "
        "major.minor release, but received supernode version 1.30.0."
    )


def test_runtime_version_metadata_tolerates_missing_metadata() -> None:
    """Missing metadata should be surfaced distinctly for rollout handling."""
    local = RuntimeVersionMetadata("flwr", "1.29.0", "superlink")

    result = local.check_compatibility_with(None)

    assert result.status == "missing"
    assert result.reason is None


def test_missing_metadata_is_tolerated_with_unknown_local_version() -> None:
    """Missing metadata should remain the rollout case in source environments."""
    local = RuntimeVersionMetadata("unknown", "unknown", "superlink")

    result = local.check_compatibility_with(None)

    assert result.status == "missing"
    assert result.reason is None


def test_version_checks_are_disabled_for_unknown_local_version() -> None:
    """Unparseable local versions should not break the receiving API."""
    local = RuntimeVersionMetadata("unknown", "unknown", "superlink")

    result = local.check_compatibility_with(
        RuntimeVersionMetadata("flwr", "1.29.0", "simulation")
    )

    assert result.status == "disabled"
    assert result.peer_metadata == RuntimeVersionMetadata(
        "flwr", "1.29.0", "simulation"
    )


def test_version_checks_are_disabled_for_unknown_peer_version() -> None:
    """Unparseable peer versions should not hard-fail rollout/dev callers."""
    local = RuntimeVersionMetadata("flwr", "1.29.0", "superlink")

    result = local.check_compatibility_with(
        RuntimeVersionMetadata("flwr", "main", "supernode")
    )

    assert result.status == "disabled"
    assert (
        result.reason
        == "Peer Flower version metadata cannot be parsed, version checks "
        "are disabled: 'main'."
    )


def test_format_invalid_metadata_message() -> None:
    """Invalid metadata messages should include the connection name."""
    assert (
        format_invalid_metadata_message(
            "CLI <-> SuperLink Control API",
            "Missing required Flower runtime metadata: flwr-component-name.",
        )
        == "Invalid Flower version metadata for CLI <-> SuperLink Control API. "
        "Missing required Flower runtime metadata: flwr-component-name."
    )
