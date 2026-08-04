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
"""Definition of one account-scoped connector."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from flwr.supercore.typing import JSONObject, JSONValue

from .oauth import OAuthConnectorProvider

ConnectorHandler = Callable[..., JSONValue]


@dataclass(frozen=True)
class ConnectorDefinition:
    """Describe all runtime components of one account-scoped connector."""

    ref: str
    tools: tuple[JSONObject, ...]
    handlers: Mapping[str, ConnectorHandler]
    oauth_provider: OAuthConnectorProvider | None = None

    def __post_init__(self) -> None:
        """Reject incomplete definitions when the connector is imported."""
        if not self.ref:
            raise ValueError("Connector reference must not be empty.")
        tool_names = [tool.get("name") for tool in self.tools]
        if not all(isinstance(name, str) and name for name in tool_names):
            raise ValueError(f"Connector '{self.ref}' has an invalid tool name.")
        if len(tool_names) != len(set(tool_names)):
            raise ValueError(f"Connector '{self.ref}' has duplicate tool names.")
        if set(tool_names) != set(self.handlers):
            raise ValueError(
                f"Connector '{self.ref}' tool schemas and handlers do not match."
            )
        if (
            self.oauth_provider is not None
            and self.oauth_provider.connector_ref != self.ref
        ):
            raise ValueError(
                f"Connector '{self.ref}' has an OAuth provider for "
                f"'{self.oauth_provider.connector_ref}'."
            )
