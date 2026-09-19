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
"""Tests for the connector registry."""


import pytest

from . import registry
from .filesystem.filesystem import FILESYSTEM_ALLOWED_DIRS_ENV
from .registry import CONNECTORS


def test_connector_references_are_unique() -> None:
    """Connector references should be unique."""
    connector_refs = [connector.ref for connector in CONNECTORS]

    assert len(connector_refs) == len(set(connector_refs))


def test_connector_tool_names_are_unique() -> None:
    """Connector tool names should be unique."""
    tool_names = [name for connector in CONNECTORS for name in connector.handlers]

    assert len(tool_names) == len(set(tool_names))


def test_filesystem_tool_hidden_without_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Filesystem tools should be unavailable when allowed dirs are unset."""
    monkeypatch.delenv(FILESYSTEM_ALLOWED_DIRS_ENV, raising=False)

    tool_names = [tool["name"] for tool in registry.get_builtin_connector_tools()]

    assert "filesystem" not in tool_names
    assert not registry.has_builtin_connector("filesystem")
    with pytest.raises(ValueError, match="not configured"):
        registry.get_connector_tools("filesystem")
    with pytest.raises(ValueError, match="Unsupported connector"):
        registry.get_builtin_connector_tool("filesystem")


def test_filesystem_tool_visible_with_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Filesystem tools should be available when allowed dirs are configured."""
    monkeypatch.setenv(FILESYSTEM_ALLOWED_DIRS_ENV, "/tmp/example")

    tool_names = [tool["name"] for tool in registry.get_builtin_connector_tools()]

    assert "filesystem" in tool_names
    assert registry.has_builtin_connector("filesystem")
    assert registry.get_connector_tools("filesystem")[0]["name"] == "filesystem"
    assert registry.get_builtin_connector_tool("filesystem")["name"] == "filesystem"
