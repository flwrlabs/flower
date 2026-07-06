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
"""Tests for SuperLink FastAPI extension hooks."""

from fastapi import FastAPI

from flwr.superlink import extensions


def test_configure_app_is_noop() -> None:
    """Test that extensions hook does not configure the app."""
    app = FastAPI()
    routes_before = list(app.routes)
    middleware_before = list(app.user_middleware)

    extensions.configure_app(app)

    assert app.routes == routes_before
    assert app.user_middleware == middleware_before


def test_get_lifespan_contexts_returns_empty_tuple() -> None:
    """Test that the extensions hook has no lifespan contexts."""
    assert not extensions.get_lifespan_contexts()
