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
"""Tests for generated provider package loading."""

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from flwr.supercore.typing import JSONObject

from ..definition import ActionAccess, ActionDefinition, ProviderDefinition
from . import loader

_PACKAGE = "flwr.supercore.task_process.connector.providers.example"


def test_provider_executors_are_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loading definitions should not import provider executor modules."""
    definition_module = ModuleType(f"{_PACKAGE}.definition")
    definition_module.__dict__["PROVIDER"] = ProviderDefinition(
        ref="example",
        display_name="Example",
        description="Example provider.",
        actions=(
            ActionDefinition(
                name="read",
                description="Read an example.",
                access=ActionAccess.READ,
                input_schema={"type": "object", "properties": {}},
            ),
        ),
    )
    executor_module = ModuleType(f"{_PACKAGE}.executors")

    def read(arguments: JSONObject, context: object) -> JSONObject:
        return {"arguments": arguments}

    executor_module.__dict__["EXECUTORS"] = {"read": read}
    monkeypatch.setattr(loader, "PROVIDER_PACKAGES", (_PACKAGE,))
    monkeypatch.setitem(sys.modules, f"{_PACKAGE}.definition", definition_module)
    connectors = loader.load_connectors()
    assert f"{_PACKAGE}.executors" not in sys.modules

    monkeypatch.setitem(sys.modules, f"{_PACKAGE}.executors", executor_module)
    result = connectors[0].handlers["example_read"]({"id": "1"}, Mock())
    assert result == {"arguments": {"id": "1"}}
    loader.validate_provider_packages()


def test_package_validation_rejects_executor_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contract validation should reject undeclared or missing executors."""
    definition_module = ModuleType(f"{_PACKAGE}.definition")
    definition_module.__dict__["PROVIDER"] = ProviderDefinition(
        ref="example",
        display_name="Example",
        description="Example provider.",
        actions=(
            ActionDefinition(
                name="read",
                description="Read an example.",
                access=ActionAccess.READ,
                input_schema={"type": "object", "properties": {}},
            ),
        ),
    )
    executor_module = ModuleType(f"{_PACKAGE}.executors")
    executor_module.__dict__["EXECUTORS"] = {}
    monkeypatch.setattr(loader, "PROVIDER_PACKAGES", (_PACKAGE,))
    monkeypatch.setitem(sys.modules, f"{_PACKAGE}.definition", definition_module)
    monkeypatch.setitem(sys.modules, f"{_PACKAGE}.executors", executor_module)

    with pytest.raises(ValueError, match="actions and executors do not match"):
        loader.validate_provider_packages()
