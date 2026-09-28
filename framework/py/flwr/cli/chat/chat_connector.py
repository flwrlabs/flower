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
"""Connector selection helpers for Flower Chat."""

from collections.abc import Iterable

import click
from prompt_toolkit.completion import Completion

from flwr.cli.constant import CHAT_CONNECTOR_COMMAND
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    Connector,
    ListConnectorsRequest,
)
from flwr.supercore.control import ControlHttpClient

from ..utils import flwr_cli_exc_handler

CHAT_CONNECTOR_CLEAR = "clear"


def fetch_chat_connectors(stub: ControlHttpClient, federation: str) -> list[Connector]:
    """Return connected connectors available in a federation."""
    with flwr_cli_exc_handler():
        response = stub.ListConnectors(ListConnectorsRequest(federation=federation))
    return [connector for connector in response.connectors if connector.connected]


def complete_connectors(
    query: str, connectors: list[Connector]
) -> Iterable[Completion]:
    """Yield connected connectors matching a completion query."""
    ref_width = max(
        len(CHAT_CONNECTOR_CLEAR),
        *(len(connector.connector_ref) for connector in connectors),
    )
    if CHAT_CONNECTOR_CLEAR.startswith(query.lower()):
        yield Completion(
            CHAT_CONNECTOR_CLEAR,
            start_position=-len(query),
            display=(
                f"{CHAT_CONNECTOR_CLEAR:<{ref_width}}        "
                "Clear selected connectors"
            ),
            selected_style="#ffffff bg:#dc8400 noreverse",
        )
    for connector in connectors:
        if connector.connector_ref.lower().startswith(query.lower()):
            yield Completion(
                connector.connector_ref,
                start_position=-len(query),
                display=(
                    f"{connector.connector_ref:<{ref_width}}        "
                    f"{connector.description}"
                ),
                selected_style="#ffffff bg:#dc8400 noreverse",
            )


def select_connector(prompt: str, connectors: list[Connector]) -> Connector:
    """Return the connector selected by a command prompt."""
    connector_ref = prompt[len(CHAT_CONNECTOR_COMMAND) :].strip()
    for connector in connectors:
        if connector.connector_ref == connector_ref:
            return connector
    raise click.ClickException(f"Unknown connector: {connector_ref}")
