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
"""Runtime types for executing declarative provider actions."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from flwr.supercore.task_process.usage import TaskUsageRecorder
from flwr.supercore.typing import JSONObject, JSONValue

from .definition import ProviderDefinition
from .http import ConnectorHttpClient
from .oauth import OAuthConnectorProvider


@dataclass(frozen=True)
class ConnectorContext:
    """Infrastructure supplied to one account-scoped action executor."""

    credentials: JSONObject
    config: JSONObject
    usage_recorder: TaskUsageRecorder
    http: ConnectorHttpClient | None


ConnectorExecutor = Callable[[JSONObject, ConnectorContext], JSONValue]


@dataclass(frozen=True)
class ConnectorDefinition:
    """Combine one provider definition with its runtime executors."""

    provider: ProviderDefinition
    executors: Mapping[str, ConnectorExecutor]
    oauth_provider: OAuthConnectorProvider | None = None

    def __post_init__(self) -> None:
        """Reject action/executor drift when a provider is loaded."""
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
                f"Provider '{self.ref}' has OAuth configuration for "
                f"'{self.oauth_provider.connector_ref}'."
            )

    @property
    def ref(self) -> str:
        """Return the provider reference."""
        return self.provider.ref

    @property
    def tools(self) -> tuple[JSONObject, ...]:
        """Return model-facing tools for this provider."""
        return tuple(action.tool(self.ref) for action in self.provider.actions)

    @property
    def handlers(self) -> Mapping[str, ConnectorExecutor]:
        """Return executors keyed by globally unique tool name."""
        return {
            action.tool_name(self.ref): self.executors[action.name]
            for action in self.provider.actions
        }
