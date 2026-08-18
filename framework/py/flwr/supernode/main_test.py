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


from unittest.mock import Mock

from fastapi.routing import iter_route_contexts

from flwr.supercore.protobuf.translation import ProtobufTranslationMiddleware
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


def test_create_app_configures_runtime_dependencies() -> None:
    """Expose the shared state and protobuf translation to Runtime routes."""
    state_factory = Mock(spec=NodeStateFactory)
    secret = b"secret"

    app = create_app(state_factory, secret)

    assert app.state.nodestate_factory is state_factory
    assert app.state.superexec_auth_secret is secret
    assert any(
        middleware.cls is ProtobufTranslationMiddleware
        for middleware in app.user_middleware
    )
