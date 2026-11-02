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
from dataclasses import replace
from types import ModuleType

import pytest

from . import loader
from .definition import ConnectorDefinition, ProviderDefinition

_PACKAGE = "flwr.supercore.task_process.connector.example"


@pytest.mark.parametrize("grouped", [False, True])
def test_load_oauth_connectors(monkeypatch: pytest.MonkeyPatch, grouped: bool) -> None:
    """Packages can export a single connector or several distinct connectors."""
    module = ModuleType(f"{_PACKAGE}.definition")
    connector = ConnectorDefinition.from_provider(
        provider=ProviderDefinition(
            ref="example",
            display_name="Example",
            description="Example connector.",
            actions=(),
        ),
        executors={},
    )
    connectors: tuple[ConnectorDefinition, ...]
    if grouped:
        connectors = (connector, replace(connector, ref="other"))
        module.__dict__["CONNECTORS"] = connectors
    else:
        connectors = (connector,)
        module.__dict__["CONNECTOR"] = connector
    monkeypatch.setattr(loader, "CONNECTOR_PACKAGES", (_PACKAGE,))
    monkeypatch.setitem(sys.modules, f"{_PACKAGE}.definition", module)

    assert loader.load_oauth_connectors() == connectors


def test_load_oauth_connectors_rejects_package_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connector reference should match its package name."""
    module = ModuleType(f"{_PACKAGE}.definition")
    module.__dict__["CONNECTOR"] = ConnectorDefinition.from_provider(
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
        loader.load_oauth_connectors()
