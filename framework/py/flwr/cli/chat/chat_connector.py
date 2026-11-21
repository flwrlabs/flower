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

from collections import Counter
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


def _connector_selection_labels(connectors: list[Connector]) -> list[str]:
    """Return human-readable labels that uniquely identify connections."""
    display_names = [
        connector.display_name or connector.connector_ref for connector in connectors
    ]
    counts = Counter(display_name.casefold() for display_name in display_names)
    return [
        (
            f"{display_name} ({connector.connector_id})"
            if counts[display_name.casefold()] > 1
            else display_name
        )
        for connector, display_name in zip(connectors, display_names, strict=True)
    ]


def fetch_chat_connectors(stub: ControlHttpClient, federation: str) -> list[Connector]:
    """Return connected connectors available in a federation."""
    with flwr_cli_exc_handler():
        response = stub.ListConnectors(ListConnectorsRequest(federation=federation))
    return [connector for connector in response.connectors if connector.connected]


def complete_connectors(
    query: str, connectors: list[Connector]
) -> Iterable[Completion]:
    """Yield connected connectors matching a completion query."""
    selection_labels = _connector_selection_labels(connectors)
    name_width = max(
        len(CHAT_CONNECTOR_CLEAR),
        *(len(selection_label) for selection_label in selection_labels),
    )
    if CHAT_CONNECTOR_CLEAR.startswith(query.lower()):
        yield Completion(
            CHAT_CONNECTOR_CLEAR,
            start_position=-len(query),
            display=(
                f"{CHAT_CONNECTOR_CLEAR:<{name_width}}        "
                "Clear selected connectors"
            ),
            selected_style="#ffffff bg:#dc8400 noreverse",
        )
    normalized_query = query.lower()
    for connector, selection_label in zip(connectors, selection_labels, strict=True):
        if (
            connector.connector_ref.lower().startswith(normalized_query)
            or selection_label.lower().startswith(normalized_query)
            or str(connector.connector_id).startswith(normalized_query)
        ):
            yield Completion(
                selection_label,
                start_position=-len(query),
                display=(
                    f"{selection_label:<{name_width}}        {connector.description}"
                ),
                selected_style="#ffffff bg:#dc8400 noreverse",
            )


def select_connector(prompt: str, connectors: list[Connector]) -> Connector:
    """Return the connector selected by a command prompt."""
    selected_label = prompt[len(CHAT_CONNECTOR_COMMAND) :].strip()
    for connector, selection_label in zip(
        connectors, _connector_selection_labels(connectors), strict=True
    ):
        if selection_label.casefold() == selected_label.casefold():
            return connector
    raise click.ClickException(f"Unknown connector: {selected_label}")
