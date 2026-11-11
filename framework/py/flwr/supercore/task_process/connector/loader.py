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
from typing import cast

from .definition import ConnectorDefinition
from .registry_generated import CONNECTOR_PACKAGES


def load_oauth_connectors() -> tuple[ConnectorDefinition, ...]:
    """Load OAuth connectors listed in the generated package registry."""
    return tuple(
        connector
        for package in CONNECTOR_PACKAGES
        for connector in _load_connectors(package)
    )


def _load_connectors(package: str) -> tuple[ConnectorDefinition, ...]:
    """Load a single connector or a group exported by a package."""
    module = import_module(f"{package}.definition")
    connectors = getattr(module, "CONNECTORS", None)
    if connectors is not None:
        if (
            not isinstance(connectors, tuple)
            or not connectors
            or any(not isinstance(item, ConnectorDefinition) for item in connectors)
        ):
            raise TypeError(
                f"Connector package '{package}' exports invalid CONNECTORS."
            )
        return cast(tuple[ConnectorDefinition, ...], connectors)

    connector = getattr(module, "CONNECTOR", None)
    if not isinstance(connector, ConnectorDefinition):
        raise TypeError(f"Connector package '{package}' does not export CONNECTOR.")
    package_ref = package.rsplit(".", maxsplit=1)[-1]
    if package_ref != connector.ref:
        raise ValueError(
            f"Connector package '{package}' must match reference '{connector.ref}'."
        )
    return (connector,)
