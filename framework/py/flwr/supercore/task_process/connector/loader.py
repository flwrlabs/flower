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
"""Load OAuth connector packages from the generated registry."""

from importlib import import_module

from .definition import ConnectorDefinition
from .registry_generated import CONNECTOR_PACKAGES


def load_oauth_connectors() -> tuple[ConnectorDefinition, ...]:
    """Load OAuth connectors listed in the generated package registry."""
    return tuple(_load_connector(package) for package in CONNECTOR_PACKAGES)


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
