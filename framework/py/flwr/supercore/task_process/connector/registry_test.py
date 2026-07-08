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

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import Mock

import pytest

from flwr.supercore.task_process.usage import TaskUsageRecorder
from flwr.supercore.typing import JSONObject, JSONValue

from . import browser_use
from . import registry as registry_module
from . import web_fetch, web_search
from .registry import (
    ConnectorDefinition,
    get_builtin_connector_tool,
    has_builtin_connector,
    invoke_connector,
)


@dataclass(frozen=True)
class _FakeToolProvider:
    definition: ConnectorDefinition

    def tool_definitions(self) -> list[JSONObject]:
        """Return one fake tool definition."""
        return [
            {
                "type": "function",
                "name": self.definition.connector_ref,
                "description": self.definition.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            }
        ]

    def execute_tool(
        self,
        *,
        arguments: JSONObject,
        usage_recorder: TaskUsageRecorder,
    ) -> JSONValue:
        """Return fake execution output."""
        return {
            "arguments": arguments,
            "usage_recorder": str(type(usage_recorder).__name__),
        }


def test_builtin_connector_tools_preserve_existing_definitions() -> None:
    """Registry-backed built-in tools should keep their existing definitions."""
    assert has_builtin_connector(web_search.WEB_SEARCH_CONNECTOR_NAME)
    assert not has_builtin_connector("slack")
    assert (
        get_builtin_connector_tool(web_search.WEB_SEARCH_CONNECTOR_NAME)
        == web_search.make_web_search_tool()
    )
    assert (
        get_builtin_connector_tool(web_fetch.WEB_FETCH_CONNECTOR_NAME)
        == web_fetch.make_web_fetch_tool()
    )
    assert (
        get_builtin_connector_tool(browser_use.BROWSER_USE_CONNECTOR_NAME)
        == browser_use.make_browser_use_tool()
    )


def test_invoke_connector_dispatches_to_registered_tool_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connector invocation should route through the full provider registry."""
    provider = _FakeToolProvider(
        definition=ConnectorDefinition(
            connector_ref="fake",
            display_name="Fake",
            description="Fake connector.",
            oauth_enabled=False,
        )
    )
    monkeypatch.setitem(registry_module._CONNECTOR_TOOL_PROVIDERS, "fake", provider)

    assert not has_builtin_connector("fake")
    with pytest.raises(ValueError, match="Unsupported connector 'fake'"):
        get_builtin_connector_tool("fake")
    assert invoke_connector(
        name="fake",
        arguments={"query": "Flower"},
        usage_recorder=Mock(spec=TaskUsageRecorder),
    ) == {
        "arguments": {"query": "Flower"},
        "usage_recorder": "Mock",
    }


def test_unknown_connector_ref_raises_value_error() -> None:
    """Unknown connector refs should keep the existing unsupported error."""
    with pytest.raises(ValueError, match="Unsupported connector 'missing'"):
        get_builtin_connector_tool("missing")

    with pytest.raises(ValueError, match="Unsupported connector 'missing'"):
        invoke_connector("missing", {}, Mock(spec=TaskUsageRecorder))
