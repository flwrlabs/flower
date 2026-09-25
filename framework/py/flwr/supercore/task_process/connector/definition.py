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
"""Definitions shared by built-in and federation-scoped connectors."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from flwr.supercore.task_process.usage import TaskUsageRecorder
from flwr.supercore.typing import JSONObject, JSONValue

from .oauth import OAuthFlow

ConnectorHandler = Callable[..., JSONValue]


class ActionAccess(StrEnum):
    """Classify whether a connector action reads or writes provider data."""

    READ = "read"
    WRITE = "write"


@dataclass(frozen=True)
class ActionDefinition:
    """Describe one provider action independently of its execution."""

    name: str
    description: str
    access: ActionAccess
    input_schema: JSONObject
    strict: bool = False

    def tool_name(self, provider_ref: str) -> str:
        """Return the globally unique model-facing action name."""
        return f"{provider_ref}_{self.name}"

    def tool(self, provider_ref: str) -> JSONObject:
        """Return the model-facing function tool for this action."""
        return {
            "type": "function",
            "name": self.tool_name(provider_ref),
            "description": self.description,
            "parameters": self.input_schema,
            "strict": self.strict,
        }


@dataclass(frozen=True)
# pylint: disable-next=too-many-instance-attributes
class OAuth2Definition:
    """Describe a standard OAuth 2 authorization-code integration."""

    authorization_url: str
    token_url: str
    client_id_env: str
    client_secret_env: str
    redirect_uri_env: str
    scopes: tuple[str, ...] = ()
    scope_parameter: str = "scope"
    scope_separator: Literal[" ", ","] = " "
    token_auth_method: Literal["client_secret_basic", "client_secret_post"] = (
        "client_secret_basic"
    )
    token_response_path: tuple[str, ...] = ()
    success_field: str | None = None
    use_pkce: bool = False
    authorization_params: Mapping[str, str] = field(default_factory=dict)
    token_request_format: Literal["form", "json"] = "form"
    token_headers: Mapping[str, str] = field(default_factory=dict)
    config_fields: tuple[str, ...] = ()
    allow_additional_scopes: bool = True
    expected_token_type: str | None = None


@dataclass(frozen=True)
class ProviderDefinition:
    """Describe one federation-scoped connector provider."""

    ref: str
    display_name: str
    description: str
    actions: tuple[ActionDefinition, ...]
    oauth: OAuth2Definition | None = None


@dataclass(frozen=True)
class ConnectorExecutionContext:
    """Infrastructure supplied to one connector action execution."""

    credentials: JSONObject
    config: JSONObject
    usage_recorder: TaskUsageRecorder


ConnectorExecutor = Callable[[JSONObject, ConnectorExecutionContext], JSONValue]


@dataclass(frozen=True)
class ConnectorDefinition:
    """Group a connector's identity, tools, execution, and optional authentication."""

    ref: str
    tools: tuple[JSONObject, ...]
    executors: Mapping[str, ConnectorExecutor]
    requires_credentials: bool = False
    provider: ProviderDefinition | None = None
    oauth_flow: OAuthFlow | None = None

    @classmethod
    def from_provider(
        cls,
        provider: ProviderDefinition,
        executors: Mapping[str, ConnectorExecutor],
        oauth_flow: OAuthFlow | None = None,
    ) -> ConnectorDefinition:
        """Build a federation-scoped connector from its provider actions."""
        action_names = {action.name for action in provider.actions}
        if action_names != set(executors):
            raise ValueError(
                f"Provider '{provider.ref}' actions and executors do not match."
            )
        return cls(
            ref=provider.ref,
            tools=tuple(action.tool(provider.ref) for action in provider.actions),
            executors={
                action.tool_name(provider.ref): executors[action.name]
                for action in provider.actions
            },
            requires_credentials=True,
            provider=provider,
            oauth_flow=oauth_flow,
        )


def build_executor(handler: ConnectorHandler) -> ConnectorExecutor:
    """Build an executor that unpacks tool arguments and forwards the context."""

    def execute(arguments: JSONObject, context: ConnectorExecutionContext) -> JSONValue:
        return handler(**arguments, context=context)

    return execute
