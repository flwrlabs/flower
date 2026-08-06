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
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum

from flwr.supercore.task_process.usage import TaskUsageRecorder
from flwr.supercore.typing import JSONObject, JSONValue

from .oauth import OAuthConnectorProvider

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

    def __post_init__(self) -> None:
        """Reject invalid action definitions."""
        if not _is_identifier(self.name) or not self.description.strip():
            raise ValueError("Connector action metadata is invalid.")
        if not isinstance(self.access, ActionAccess):
            raise ValueError(f"Action '{self.name}' has invalid access metadata.")
        if self.input_schema.get("type") != "object":
            raise ValueError(f"Action '{self.name}' input schema must be an object.")

    def tool_name(self, provider_ref: str) -> str:
        """Return the globally unique model-facing action name."""
        return f"{provider_ref}_{self.name}"

    def tool(self, provider_ref: str) -> JSONObject:
        """Return the model-facing function tool for this action."""
        return {
            "type": "function",
            "name": self.tool_name(provider_ref),
            "description": self.description,
            "parameters": deepcopy(self.input_schema),
        }


@dataclass(frozen=True)
class ProviderDefinition:
    """Describe one account-scoped connector provider."""

    ref: str
    display_name: str
    description: str
    actions: tuple[ActionDefinition, ...]

    def __post_init__(self) -> None:
        """Reject incomplete provider definitions."""
        if (
            not _is_identifier(self.ref)
            or not self.display_name.strip()
            or not self.description.strip()
        ):
            raise ValueError("Connector provider metadata must not be empty.")
        names = [action.name for action in self.actions]
        if not names or len(names) != len(set(names)):
            raise ValueError(f"Provider '{self.ref}' has invalid action names.")


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
    oauth_provider: OAuthConnectorProvider | None = None

    def __post_init__(self) -> None:
        """Reject incomplete definitions when the connector is imported."""
        action_names = {action.name for action in self.provider.actions}
        if action_names != set(self.executors):
            raise ValueError(
                f"Provider '{self.ref}' actions and executors do not match."
            )
        if (
            self.oauth_provider is not None
            and self.oauth_provider.connector_ref != self.ref
        ):
            raise ValueError(
                f"Connector '{self.ref}' has an OAuth provider for "
                f"'{self.oauth_provider.connector_ref}'."
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


def _is_identifier(value: str) -> bool:
    """Return whether a value is a lowercase connector identifier."""
    return bool(value) and value.isidentifier() and value.lower() == value
