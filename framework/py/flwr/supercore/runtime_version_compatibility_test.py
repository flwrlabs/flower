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

from .runtime_version_compatibility import (
    RuntimeVersionMetadata,
    format_invalid_metadata_message,
    get_runtime_version_rejection,
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
            "Missing required Flower runtime metadata: "
            "flwr-component-name, flwr-package-version.",
        ),
        (
            (
                (FLWR_PACKAGE_NAME_METADATA_KEY, b"flwr"),
                (FLWR_PACKAGE_VERSION_METADATA_KEY, b"1.29.0"),
                (FLWR_COMPONENT_NAME_METADATA_KEY, b"cli"),
            ),
            "Flower runtime metadata contains non-string values: "
            "flwr-component-name, flwr-package-name, flwr-package-version.",
        ),
        (
            (
                (FLWR_PACKAGE_NAME_METADATA_KEY, "flwr"),
                (FLWR_PACKAGE_VERSION_METADATA_KEY, b"\xff\xfe"),
                (FLWR_COMPONENT_NAME_METADATA_KEY, "cli"),
            ),
            "Flower runtime metadata contains non-string values: "
            "flwr-package-version.",
        ),
        (
            (
                (FLWR_PACKAGE_NAME_METADATA_KEY, "flwr"),
                (FLWR_PACKAGE_VERSION_METADATA_KEY, "1.29.0"),
                (FLWR_PACKAGE_VERSION_METADATA_KEY, "1.29.1"),
                (FLWR_COMPONENT_NAME_METADATA_KEY, "cli"),
            ),
            "Flower runtime metadata contains duplicate values: "
            "flwr-package-version.",
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
    rejection = get_runtime_version_rejection(
        "SuperNode <-> SuperLink Fleet API",
        local,
        peer,
    )

    assert rejection is None


@pytest.mark.parametrize(
    ("local", "peer", "connection_name", "expected_rejection"),
    [
        (
            RuntimeVersionMetadata("flwr", "1.29.2", "superlink"),
            RuntimeVersionMetadata("flwr", "1.30.0", "supernode"),
            "SuperNode <-> SuperLink Fleet API",
            "Incompatible Flower version for SuperNode <-> SuperLink Fleet API.\n"
            "Local superlink version 1.29.2 only accepts peers from the same "
            "major.minor release, but received supernode version 1.30.0.",
        ),
        (
            RuntimeVersionMetadata("unknown", "unknown", "superlink"),
            RuntimeVersionMetadata("flwr", "1.29.0", "simulation"),
            "ServerApp <-> SuperLink ServerAppIo API",
            "Invalid Flower version metadata for "
            "ServerApp <-> SuperLink ServerAppIo API. "
            "Local Flower package name is not recognized: 'unknown'.",
        ),
        (
            RuntimeVersionMetadata("flwr", "1.29.0", "superlink"),
            RuntimeVersionMetadata("flwr", "main", "supernode"),
            "SuperNode <-> SuperLink Fleet API",
            "Invalid Flower version metadata for "
            "SuperNode <-> SuperLink Fleet API. "
            "Peer Flower version metadata cannot be parsed: 'main'.",
        ),
        (
            RuntimeVersionMetadata("flwr", "1.29.0", "superlink"),
            RuntimeVersionMetadata("forked-flower", "1.29.1", "supernode"),
            "SuperNode <-> SuperLink Fleet API",
            "Invalid Flower version metadata for "
            "SuperNode <-> SuperLink Fleet API. "
            "Peer Flower package name is not recognized: 'forked-flower'.",
        ),
    ],
)
def test_runtime_version_metadata_rejects_expected_cases(
    local: RuntimeVersionMetadata,
    peer: RuntimeVersionMetadata,
    connection_name: str,
    expected_rejection: str,
) -> None:
    """Explicitly invalid or incompatible peers should be rejected."""
    assert (
        get_runtime_version_rejection(connection_name, local, peer)
        == expected_rejection
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
