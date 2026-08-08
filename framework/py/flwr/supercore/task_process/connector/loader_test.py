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
"""Tests for generated connector loading."""

import sys
from types import ModuleType

import pytest

from . import loader
from .definition import ConnectorDefinition, ProviderDefinition

_PACKAGE = "flwr.supercore.task_process.connector.example"


def test_load_connectors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Registered definition modules should provide connector definitions."""
    module = ModuleType(f"{_PACKAGE}.definition")
    connector = ConnectorDefinition(
        provider=ProviderDefinition(
            ref="example",
            display_name="Example",
            description="Example connector.",
            actions=(),
        ),
        executors={},
    )
    module.__dict__["CONNECTOR"] = connector
    monkeypatch.setattr(loader, "CONNECTOR_PACKAGES", (_PACKAGE,))
    monkeypatch.setitem(sys.modules, f"{_PACKAGE}.definition", module)

    assert loader.load_connectors() == (connector,)


def test_load_connectors_rejects_package_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connector reference should match its package name."""
    module = ModuleType(f"{_PACKAGE}.definition")
    module.__dict__["CONNECTOR"] = ConnectorDefinition(
        provider=ProviderDefinition(
            ref="other",
            display_name="Other",
            description="Other connector.",
            actions=(),
        ),
        executors={},
    )
    monkeypatch.setattr(loader, "CONNECTOR_PACKAGES", (_PACKAGE,))
    monkeypatch.setitem(sys.modules, f"{_PACKAGE}.definition", module)

    with pytest.raises(ValueError, match="must match reference"):
        loader.load_connectors()
