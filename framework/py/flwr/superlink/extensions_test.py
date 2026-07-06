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

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from types import ModuleType
from typing import Any

import pytest
from fastapi import FastAPI

from flwr.superlink import extensions


def _mock_ee_extensions(monkeypatch: pytest.MonkeyPatch, module: ModuleType) -> None:
    def import_module_mock(name: str) -> ModuleType:
        assert name == extensions.EE_EXTENSIONS_MODULE
        return module

    monkeypatch.setattr(extensions, "import_module", import_module_mock)


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


def test_configure_app_delegates_to_ee_extensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that app configuration delegates to EE extensions."""
    app = FastAPI()
    configured_apps: list[FastAPI] = []
    module = ModuleType(extensions.EE_EXTENSIONS_MODULE)

    def configure_ee_app(fastapi_app: FastAPI) -> None:
        configured_apps.append(fastapi_app)

    setattr(module, "configure_app", configure_ee_app)
    _mock_ee_extensions(monkeypatch, module)

    extensions.configure_app(app)

    assert configured_apps == [app]


def test_get_lifespan_contexts_delegates_to_ee_extensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that lifespan contexts delegate to EE extensions."""
    module = ModuleType(extensions.EE_EXTENSIONS_MODULE)

    @asynccontextmanager
    async def lifespan_context(
        _: FastAPI,
    ) -> AsyncIterator[Mapping[str, Any] | None]:
        yield {"extension": True}

    def get_ee_lifespan_contexts() -> tuple[extensions.SuperLinkLifespanContext, ...]:
        return (lifespan_context,)

    setattr(module, "get_lifespan_contexts", get_ee_lifespan_contexts)
    _mock_ee_extensions(monkeypatch, module)

    assert extensions.get_lifespan_contexts() == (lifespan_context,)
