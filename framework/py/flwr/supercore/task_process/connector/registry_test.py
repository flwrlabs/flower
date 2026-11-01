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

from unittest.mock import Mock, patch

from . import registry
from .registry import CONNECTORS


def test_connector_references_are_unique() -> None:
    """Connector references should be unique."""
    connector_refs = [connector.ref for connector in CONNECTORS]

    assert len(connector_refs) == len(set(connector_refs))


def test_connector_tool_names_are_unique() -> None:
    """Connector tool names should be unique."""
    tool_names = [name for connector in CONNECTORS for name in connector.executors]

    assert len(tool_names) == len(set(tool_names))


def test_dynamic_tools_resolve_to_their_connector() -> None:
    """Resolve and directly execute tools on their credential-backed connector."""
    connector = Mock(ref="example", tool_source=Mock(), requires_credentials=True)
    connector.executors = {}
    with patch.dict(
        registry._CONNECTORS_BY_REF,  # pylint: disable=protected-access
        {"example": connector},
    ):
        assert registry.get_connector_ref("example_read") == "example"
        assert registry.requires_connector_credentials("example_read")
        assert registry.is_dynamic_connector_tool("example_read")
        operation = registry.get_connector_discovery_tool("example")
        assert operation is not None
        assert not registry.is_dynamic_connector_tool(operation)
        registry.invoke_connector("example_Read", {}, Mock(), {}, {})
        assert connector.tool_source.execute.call_args.args[1].action_name == "Read"
        assert registry.get_connector_ref("exampleish_read") == "exampleish_read"
