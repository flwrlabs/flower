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
"""Tests for connector definitions."""


import pytest

from flwr.supercore.typing import JSONObject

from .definition import ConnectorDefinition
from .tool_schema import function_tool


def _handler() -> JSONObject:
    return {}


def test_connector_definition_rejects_handler_drift() -> None:
    """Every tool should have exactly one matching handler."""
    tool = function_tool("example_read", "Read an example.", properties={})
    ConnectorDefinition(
        ref="example", tools=(tool,), handlers={"example_read": _handler}
    )

    with pytest.raises(ValueError, match="schemas and handlers do not match"):
        ConnectorDefinition(ref="example", tools=(tool,), handlers={})
