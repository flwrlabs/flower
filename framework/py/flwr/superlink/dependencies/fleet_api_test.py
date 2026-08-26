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

from fastapi import FastAPI, Request

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


def test_get_fleet_api_type_returns_none_when_unconfigured() -> None:
    """Return None when the application has no Fleet API configuration."""
    assert get_fleet_api_type(_make_request(FastAPI())) is None
