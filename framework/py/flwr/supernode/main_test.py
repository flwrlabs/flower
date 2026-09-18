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
"""Tests for SuperNode FastAPI application construction."""

from fastapi.routing import iter_route_contexts
from fastapi.testclient import TestClient

from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supernode.nodestate import NodeStateFactory

from .main import create_app


def test_create_app_mounts_health_without_readiness() -> None:
    """Expose the shared health route without a SuperNode readiness route."""
    app = create_app()
    paths = {
        route_context.path_format
        for route_context in iter_route_contexts(app.routes)
        if route_context.path_format is not None
    }

    assert "/health" in paths
    assert "/ready" not in paths


def test_create_app_mounts_responses_router() -> None:
    """Expose the shared Responses endpoint through the SuperNode API."""
    app = create_app(NodeStateFactory(ObjectStoreFactory()))

    response = TestClient(app).post(
        "/v1/runtime/responses",
        json={"model": "model", "input": "hello"},
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"
