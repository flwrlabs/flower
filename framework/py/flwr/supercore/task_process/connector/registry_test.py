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

from .registry import CONNECTORS, get_connector_ref, requires_connector_credentials


def test_connector_references_are_unique() -> None:
    """Connector references should be unique."""
    connector_refs = [connector.ref for connector in CONNECTORS]

    assert len(connector_refs) == len(set(connector_refs))


def test_connector_tool_names_are_unique() -> None:
    """Connector tool names should be unique."""
    tool_names = [name for connector in CONNECTORS for name in connector.executors]

    assert len(tool_names) == len(set(tool_names))


def test_dynamic_github_tools_require_github_credentials() -> None:
    """Resolve newly discovered names without treating them as built-in tools."""
    assert get_connector_ref("github_future_tool") == "github"
    assert requires_connector_credentials("github_future_tool")
    assert get_connector_ref("githubish_future_tool") == "githubish_future_tool"
    assert not requires_connector_credentials("githubish_future_tool")
