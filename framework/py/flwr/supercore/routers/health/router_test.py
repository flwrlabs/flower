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
"""Tests for the health API router."""


from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from .router import router


def test_health_supports_get_and_head() -> None:
    """Return a successful empty response for health checks."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    for method in ("GET", "HEAD"):
        response = client.request(method, "/health")

        assert response.status_code == status.HTTP_200_OK
        assert response.content == b""


def test_router_does_not_expose_readiness() -> None:
    """Leave service-specific readiness checks to their owning application."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    assert client.get("/ready").status_code == status.HTTP_404_NOT_FOUND
