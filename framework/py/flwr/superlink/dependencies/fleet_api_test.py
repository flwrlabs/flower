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
"""Tests for the Fleet API transport type dependency."""

import pytest
from fastapi import FastAPI, Request

from flwr.supercore.error import ApiErrorCode, FlowerError

from .fleet_api import get_fleet_api_type


def _make_request(app: FastAPI) -> Request:
    """Return a minimal request bound to an application."""
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
            "query_string": b"",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "scheme": "http",
            "app": app,
        }
    )


def test_get_fleet_api_type_returns_configured_value() -> None:
    """Return the Fleet API transport type exposed by the application."""
    app = FastAPI()
    app.state.fleet_api_type = "grpc-rere"

    assert get_fleet_api_type(_make_request(app)) == "grpc-rere"


@pytest.mark.parametrize("fleet_api_type", [None, ""])
def test_get_fleet_api_type_raises_when_unconfigured(
    fleet_api_type: str | None,
) -> None:
    """Raise when the application has no valid Fleet API configuration."""
    app = FastAPI()
    app.state.fleet_api_type = fleet_api_type

    with pytest.raises(FlowerError) as exc_info:
        get_fleet_api_type(_make_request(app))

    assert exc_info.value.code == ApiErrorCode.FLEET_API_TYPE_NOT_INITIALIZED
    assert exc_info.value.message == "SuperLink Fleet API type is not initialized."
