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
"""Load generated provider definitions and lazy action executors."""

from collections.abc import Mapping
from importlib import import_module
from types import ModuleType
from typing import cast

from flwr.supercore.typing import JSONObject, JSONValue

from ..definition import ProviderDefinition
from ..oauth import load_declarative_oauth_provider
from ..runtime import ConnectorContext, ConnectorDefinition, ConnectorExecutor
from .registry_generated import PROVIDER_PACKAGES


def load_connectors() -> tuple[ConnectorDefinition, ...]:
    """Load provider definitions while keeping executor modules lazy."""
    connectors: list[ConnectorDefinition] = []
    for package in PROVIDER_PACKAGES:
        provider = _load_provider(package)
        executors = {
            action.name: _lazy_executor(package, action.name)
            for action in provider.actions
        }
        connectors.append(
            ConnectorDefinition(
                provider=provider,
                executors=executors,
                oauth_provider=load_declarative_oauth_provider(provider),
            )
        )
    return tuple(connectors)


def validate_provider_packages() -> None:
    """Import every provider package and validate executor correspondence."""
    definitions = [_load_provider(package) for package in PROVIDER_PACKAGES]
    refs = [provider.ref for provider in definitions]
    if len(refs) != len(set(refs)):
        raise ValueError("Provider references must be globally unique.")
    tool_names = [
        action.tool_name(provider.ref)
        for provider in definitions
        for action in provider.actions
    ]
    if len(tool_names) != len(set(tool_names)):
        raise ValueError("Provider action tool names must be globally unique.")
    for package, provider in zip(PROVIDER_PACKAGES, definitions, strict=True):
        executors = _load_executors(package)
        if set(executors) != {action.name for action in provider.actions}:
            raise ValueError(
                f"Provider '{provider.ref}' actions and executors do not match."
            )


def _load_provider(package: str) -> ProviderDefinition:
    """Load and validate one provider's definition module."""
    module = import_module(f"{package}.definition")
    provider = getattr(module, "PROVIDER", None)
    if not isinstance(provider, ProviderDefinition):
        raise TypeError(f"Provider package '{package}' does not export PROVIDER.")
    if package.rsplit(".", maxsplit=1)[-1] != provider.ref:
        raise ValueError(
            f"Provider package '{package}' must match reference '{provider.ref}'."
        )
    return provider


def _load_executors(package: str) -> Mapping[str, ConnectorExecutor]:
    """Load and validate one provider's executor mapping."""
    module: ModuleType = import_module(f"{package}.executors")
    executors = getattr(module, "EXECUTORS", None)
    if not isinstance(executors, Mapping):
        raise TypeError(f"Provider package '{package}' does not export EXECUTORS.")
    if not all(
        isinstance(name, str) and callable(value) for name, value in executors.items()
    ):
        raise TypeError(f"Provider package '{package}' has invalid executors.")
    return cast(Mapping[str, ConnectorExecutor], executors)


def _lazy_executor(package: str, action_name: str) -> ConnectorExecutor:
    """Return an executor which imports provider implementation on first use."""

    def execute(arguments: JSONObject, context: ConnectorContext) -> JSONValue:
        executor = _load_executors(package).get(action_name)
        if executor is None:
            raise RuntimeError(
                f"Provider action '{package}.{action_name}' has no executor."
            )
        return executor(arguments, context)

    return execute
