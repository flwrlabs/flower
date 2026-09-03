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
"""Definitions for account-scoped connectors."""

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
    output_schema: JSONObject | None = None

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
    """Describe one account-scoped connector provider."""

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
    """Combine one provider definition with its action executors."""

    provider: ProviderDefinition
    executors: Mapping[str, ConnectorExecutor]
    oauth_flow: OAuthFlow | None = None

    def __post_init__(self) -> None:
        """Reject incomplete definitions when the connector is imported."""
        action_names = {action.name for action in self.provider.actions}
        if action_names != set(self.executors):
            raise ValueError(
                f"Provider '{self.ref}' actions and executors do not match."
            )

    @property
    def ref(self) -> str:
        """Return the provider reference."""
        return self.provider.ref

    @property
    def tools(self) -> tuple[JSONObject, ...]:
        """Return model-facing tools for this connector."""
        return tuple(action.tool(self.ref) for action in self.provider.actions)

    @property
    def handlers(self) -> Mapping[str, ConnectorExecutor]:
        """Return executors keyed by globally unique tool name."""
        return {
            action.tool_name(self.ref): self.executors[action.name]
            for action in self.provider.actions
        }
