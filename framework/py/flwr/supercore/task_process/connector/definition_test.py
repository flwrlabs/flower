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
"""Tests for declarative connector definitions."""

import pytest

from .definition import (
    ActionAccess,
    ActionDefinition,
    OAuth2Definition,
    ProviderDefinition,
)
from .runtime import ConnectorDefinition


def _action(*, scopes: tuple[str, ...] = ()) -> ActionDefinition:
    return ActionDefinition(
        name="read",
        description="Read an example.",
        access=ActionAccess.READ,
        input_schema={"type": "object", "properties": {}},
        required_scopes=scopes,
    )


def _provider(*, scopes: tuple[str, ...] = ()) -> ProviderDefinition:
    return ProviderDefinition(
        ref="example",
        display_name="Example",
        description="Example provider.",
        actions=(_action(scopes=scopes),),
        oauth=OAuth2Definition(
            authorization_url="https://example.com/authorize",
            token_url="https://example.com/token",
            client_id_env="EXAMPLE_CLIENT_ID",
            client_secret_env="EXAMPLE_CLIENT_SECRET",
            redirect_uri_env="EXAMPLE_REDIRECT_URI",
            scopes=scopes,
        ),
    )


def test_provider_definition_builds_tools() -> None:
    """Provider actions should generate globally unique function tools."""
    connector = ConnectorDefinition(
        provider=_provider(scopes=("read:items",)),
        executors={"read": lambda arguments, context: {}},
    )

    assert connector.tools == (
        {
            "type": "function",
            "name": "example_read",
            "description": "Read an example.",
            "parameters": {"type": "object", "properties": {}},
        },
    )
    assert connector.provider.actions[0].access is ActionAccess.READ


def test_definitions_reject_drift() -> None:
    """Definitions should reject undeclared scopes and missing executors."""
    with pytest.raises(ValueError, match="undeclared OAuth scopes"):
        _provider(scopes=("read:items",)).__class__(
            ref="example",
            display_name="Example",
            description="Example provider.",
            actions=(_action(scopes=("write:items",)),),
            oauth=_provider(scopes=("read:items",)).oauth,
        )
    with pytest.raises(ValueError, match="actions and executors do not match"):
        ConnectorDefinition(provider=_provider(), executors={})
