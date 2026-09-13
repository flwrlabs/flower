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
"""Load connector packages from the generated registry."""

from importlib import import_module

from .definition import ConnectorDefinition
from .registry_generated import CONNECTOR_PACKAGES


def load_connectors() -> tuple[ConnectorDefinition, ...]:
    """Load registered connectors and validate global identifiers."""
    connectors = tuple(_load_connector(package) for package in CONNECTOR_PACKAGES)
    refs = [connector.ref for connector in connectors]
    if len(refs) != len(set(refs)):
        raise ValueError("Connector references must be globally unique.")
    tool_names = [name for connector in connectors for name in connector.handlers]
    if len(tool_names) != len(set(tool_names)):
        raise ValueError("Connector tool names must be globally unique.")
    return connectors


def _load_connector(package: str) -> ConnectorDefinition:
    """Load one connector definition and validate its package name."""
    module = import_module(f"{package}.definition")
    connector = getattr(module, "CONNECTOR", None)
    if not isinstance(connector, ConnectorDefinition):
        raise TypeError(f"Connector package '{package}' does not export CONNECTOR.")
    package_ref = package.rsplit(".", maxsplit=1)[-1]
    if package_ref != connector.ref:
        raise ValueError(
            f"Connector package '{package}' must match reference '{connector.ref}'."
        )
    return connector
