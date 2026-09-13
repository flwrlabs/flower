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
"""Tests for the FastAPI Runtime version dependency."""

from unittest.mock import patch

import pytest
from fastapi import Request

from flwr.supercore.constant import (
    FLWR_COMPONENT_NAME_METADATA_KEY,
    FLWR_PACKAGE_NAME_METADATA_KEY,
    FLWR_PACKAGE_VERSION_METADATA_KEY,
)
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.runtime_version_compatibility import RuntimeVersionMetadata

from .runtime_version import RuntimeVersionDependency


def _make_request(headers: list[tuple[str, str]]) -> Request:
    """Return a minimal HTTP request with the given headers."""
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/runtime/claim-task",
            "headers": [(key.encode(), value.encode()) for key, value in headers],
            "query_string": b"",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "scheme": "http",
        }
    )


def _dependency() -> RuntimeVersionDependency:
    """Return a dependency with deterministic local metadata."""
    with patch.object(
        RuntimeVersionMetadata,
        "from_local_component",
        return_value=RuntimeVersionMetadata("flwr", "1.30.0", "SuperLink"),
    ):
        return RuntimeVersionDependency(
            component_name="SuperLink",
            connection_name="Caller <-> SuperLink Runtime API",
        )


@pytest.mark.parametrize(
    "headers",
    [
        [],
        [
            (FLWR_PACKAGE_NAME_METADATA_KEY, "flwr"),
            (FLWR_PACKAGE_VERSION_METADATA_KEY, "1.30.7"),
            (FLWR_COMPONENT_NAME_METADATA_KEY, "SuperExec"),
        ],
    ],
)
def test_runtime_version_dependency_accepts_expected_metadata(
    headers: list[tuple[str, str]],
) -> None:
    """Missing metadata and matching major.minor versions should be accepted."""
    _dependency()(_make_request(headers))


@pytest.mark.parametrize(
    ("headers", "expected_details"),
    [
        (
            [(FLWR_PACKAGE_NAME_METADATA_KEY, "flwr")],
            "Invalid Flower runtime metadata",
        ),
        (
            [
                (FLWR_PACKAGE_NAME_METADATA_KEY, "flwr"),
                (FLWR_PACKAGE_VERSION_METADATA_KEY, "1.29.0"),
                (FLWR_COMPONENT_NAME_METADATA_KEY, "SuperExec"),
            ],
            "runtime version mismatch",
        ),
    ],
)
def test_runtime_version_dependency_rejects_invalid_metadata(
    headers: list[tuple[str, str]], expected_details: str
) -> None:
    """Partial and incompatible Runtime metadata should be rejected."""
    with pytest.raises(FlowerError) as exc_info:
        _dependency()(_make_request(headers))

    assert exc_info.value.code == ApiErrorCode.RUNTIME_VERSION_INCOMPATIBLE
    assert expected_details in (exc_info.value.public_details or "")
