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
"""Declarative definitions for account-scoped connector providers."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Literal

from flwr.supercore.typing import JSONObject


class ActionAccess(StrEnum):
    """Provider-independent classification of an action's side effects."""

    READ = "read"
    WRITE = "write"


@dataclass(frozen=True)
class ActionDefinition:
    """Describe one provider action independently of its executor."""

    name: str
    description: str
    access: ActionAccess
    input_schema: JSONObject
    output_schema: JSONObject | None = None
    required_scopes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject invalid action definitions when a provider is imported."""
        _validate_identifier(self.name, "Action name")
        if not self.description.strip():
            raise ValueError(f"Action '{self.name}' description must not be empty.")
        if not isinstance(self.access, ActionAccess):
            raise ValueError(f"Action '{self.name}' has an invalid access class.")
        if self.input_schema.get("type") != "object":
            raise ValueError(f"Action '{self.name}' input schema must be an object.")
        if len(self.required_scopes) != len(set(self.required_scopes)):
            raise ValueError(f"Action '{self.name}' has duplicate OAuth scopes.")

    def tool(self, provider_ref: str) -> JSONObject:
        """Return the model-facing function tool for this action."""
        return {
            "type": "function",
            "name": self.tool_name(provider_ref),
            "description": self.description,
            "parameters": deepcopy(self.input_schema),
        }

    def tool_name(self, provider_ref: str) -> str:
        """Return this action's globally unique model-facing name."""
        return f"{provider_ref}_{self.name}"


@dataclass(frozen=True)
# pylint: disable-next=too-many-instance-attributes
class OAuth2Definition:
    """Describe a standard OAuth 2 authorization-code integration."""

    authorization_url: str
    token_url: str
    client_id_env: str
    client_secret_env: str | None
    redirect_uri_env: str
    scopes: tuple[str, ...] = ()
    scope_separator: Literal[" ", ","] = " "
    token_auth_method: Literal["client_secret_basic", "client_secret_post", "none"] = (
        "client_secret_basic"
    )
    token_request_format: Literal["form", "json"] = "form"
    use_pkce: bool = False
    authorization_params: JSONObject = field(default_factory=dict)
    token_params: JSONObject = field(default_factory=dict)
    token_response_path: tuple[str, ...] = ()
    config_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject incomplete OAuth definitions."""
        required = (
            self.authorization_url,
            self.token_url,
            self.client_id_env,
            self.redirect_uri_env,
        )
        if not all(value.strip() for value in required):
            raise ValueError("OAuth definition fields must not be empty.")
        if self.token_auth_method != "none" and not self.client_secret_env:
            raise ValueError("Confidential OAuth clients require a client secret.")
        if len(self.scopes) != len(set(self.scopes)):
            raise ValueError("OAuth definition has duplicate scopes.")
        object.__setattr__(
            self, "authorization_params", MappingProxyType(self.authorization_params)
        )
        object.__setattr__(self, "token_params", MappingProxyType(self.token_params))


@dataclass(frozen=True)
class ProviderDefinition:
    """Describe one account-scoped connector provider."""

    ref: str
    display_name: str
    description: str
    actions: tuple[ActionDefinition, ...]
    oauth: OAuth2Definition | None = None
    api_base_url: str | None = None
    api_headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject inconsistent provider packages when imported."""
        _validate_identifier(self.ref, "Provider reference")
        if not self.display_name.strip() or not self.description.strip():
            raise ValueError(f"Provider '{self.ref}' metadata must not be empty.")
        if not self.actions:
            raise ValueError(f"Provider '{self.ref}' must define at least one action.")
        names = [action.name for action in self.actions]
        if len(names) != len(set(names)):
            raise ValueError(f"Provider '{self.ref}' has duplicate action names.")
        if self.oauth is not None:
            missing_scopes = {
                scope
                for action in self.actions
                for scope in action.required_scopes
                if scope not in self.oauth.scopes
            }
            if missing_scopes:
                raise ValueError(
                    f"Provider '{self.ref}' actions require undeclared OAuth scopes: "
                    + ", ".join(sorted(missing_scopes))
                    + "."
                )
        object.__setattr__(self, "api_headers", MappingProxyType(self.api_headers))


def _validate_identifier(value: str, label: str) -> None:
    """Require a stable lowercase snake-case identifier."""
    if not value or not value.isidentifier() or value.lower() != value:
        raise ValueError(f"{label} must be a lowercase snake-case identifier.")
