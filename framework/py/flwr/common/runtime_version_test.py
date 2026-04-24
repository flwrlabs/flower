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

from flwr.common.constant import (
    FLWR_COMPONENT_NAME_METADATA_KEY,
    FLWR_PACKAGE_NAME_METADATA_KEY,
    FLWR_PACKAGE_VERSION_METADATA_KEY,
)

from .runtime_version import (
    ParsedFlowerVersion,
    RuntimeVersionMetadata,
    build_runtime_version_metadata,
    evaluate_runtime_version_compatibility,
    format_incompatible_version_message,
    format_invalid_metadata_message,
    parse_flower_version,
    read_runtime_version_metadata,
    runtime_version_metadata_to_dict,
)


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("1.29.0", ParsedFlowerVersion(1, 29, 0)),
        ("1.29.7", ParsedFlowerVersion(1, 29, 7)),
        ("1.29.0.dev12", ParsedFlowerVersion(1, 29, 0)),
        ("1.29.0-nightly.20260423", ParsedFlowerVersion(1, 29, 0)),
    ],
)
def test_parse_flower_version_accepts_valid_prefixes(
    version: str, expected: ParsedFlowerVersion
) -> None:
    """Leading `major.minor.patch` should be parsed consistently."""
    assert parse_flower_version(version) == expected


@pytest.mark.parametrize("version", ["1.29", "main", "dev", "unknown", "1.29.x"])
def test_parse_flower_version_rejects_invalid_values(version: str) -> None:
    """Versions without a parseable leading release tuple should fail."""
    assert parse_flower_version(version) is None


def test_runtime_version_metadata_round_trip() -> None:
    """Metadata should serialize using the shared key names."""
    metadata = build_runtime_version_metadata(
        "supernode",
        package_name_value="flwr",
        package_version_value="1.29.0",
    )

    assert runtime_version_metadata_to_dict(metadata) == {
        FLWR_PACKAGE_NAME_METADATA_KEY: "flwr",
        FLWR_PACKAGE_VERSION_METADATA_KEY: "1.29.0",
        FLWR_COMPONENT_NAME_METADATA_KEY: "supernode",
    }


def test_build_runtime_version_metadata_rejects_empty_component_name() -> None:
    """Component names must not be empty."""
    with pytest.raises(ValueError, match="component_name"):
        build_runtime_version_metadata("")


def test_read_runtime_version_metadata_returns_missing_for_absent_keys() -> None:
    """Absent Flower metadata should be treated as the rollout missing case."""
    metadata, error = read_runtime_version_metadata({"other-header": "value"})

    assert metadata is None
    assert error is None


def test_read_runtime_version_metadata_rejects_partial_metadata() -> None:
    """Partial Flower metadata should be rejected as invalid."""
    metadata, error = read_runtime_version_metadata(
        {FLWR_PACKAGE_NAME_METADATA_KEY: "flwr"}
    )

    assert metadata is None
    assert error is not None
    assert "Missing required Flower runtime metadata" in error


def test_read_runtime_version_metadata_accepts_metadata_item_iterables() -> None:
    """gRPC metadata-style iterables should be supported directly."""
    metadata, error = read_runtime_version_metadata(
        [
            (FLWR_PACKAGE_NAME_METADATA_KEY, "flwr"),
            (FLWR_PACKAGE_VERSION_METADATA_KEY, "1.29.0"),
            (FLWR_COMPONENT_NAME_METADATA_KEY, "cli"),
        ]
    )

    assert error is None
    assert metadata == RuntimeVersionMetadata(
        package_name="flwr",
        package_version="1.29.0",
        component_name="cli",
    )


def test_evaluate_runtime_version_compatibility_accepts_same_major_minor() -> None:
    """Patch differences are compatible."""
    result = evaluate_runtime_version_compatibility(
        RuntimeVersionMetadata("flwr", "1.29.0", "superlink"),
        RuntimeVersionMetadata("flwr", "1.29.7", "supernode"),
    )

    assert result.status == "compatible"
    assert result.peer_version == ParsedFlowerVersion(1, 29, 7)


def test_evaluate_runtime_version_compatibility_rejects_different_minor() -> None:
    """Different minor versions should be incompatible."""
    peer = RuntimeVersionMetadata("flwr", "1.30.0", "supernode")
    local = RuntimeVersionMetadata("flwr", "1.29.2", "superlink")

    result = evaluate_runtime_version_compatibility(local, peer)

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


def test_evaluate_runtime_version_compatibility_tolerates_missing_metadata() -> None:
    """Missing metadata should be surfaced distinctly for rollout handling."""
    result = evaluate_runtime_version_compatibility(
        RuntimeVersionMetadata("flwr", "1.29.0", "superlink"),
        None,
    )

    assert result.status == "missing"
    assert result.reason is None


def test_evaluate_runtime_version_compatibility_tolerates_missing_metadata_with_unknown_local_version() -> None:
    """Missing metadata should remain the rollout case in source environments."""
    result = evaluate_runtime_version_compatibility(
        RuntimeVersionMetadata("unknown", "unknown", "superlink"),
        None,
    )

    assert result.status == "missing"
    assert result.reason is None


def test_evaluate_runtime_version_compatibility_rejects_invalid_peer_version() -> None:
    """Unparseable peer versions should be classified as invalid metadata."""
    result = evaluate_runtime_version_compatibility(
        RuntimeVersionMetadata("flwr", "1.29.0", "superlink"),
        RuntimeVersionMetadata("flwr", "main", "supernode"),
    )

    assert result.status == "invalid"
    assert result.reason == "Peer Flower version metadata is invalid: 'main'."


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
