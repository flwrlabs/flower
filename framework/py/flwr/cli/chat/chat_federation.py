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
"""Federation selection helpers for Flower Chat."""

from collections.abc import Iterable

import click
from prompt_toolkit.completion import Completion

from flwr.cli.constant import CHAT_DEFAULT_FEDERATION_NAME, CHAT_FEDERATION_COMMAND
from flwr.proto.federation_pb2 import Federation  # pylint: disable=E0611
from flwr.supercore.constant import DEFAULT_FEDERATION_SIMULATION


def complete_federations(
    query: str, federations: list[Federation]
) -> Iterable[Completion]:
    """Yield federations matching a completion query."""
    name_width = max((len(federation.name) for federation in federations), default=0)
    for federation in federations:
        if federation.name.lower().startswith(query.lower()):
            yield Completion(
                federation.name,
                start_position=-len(query),
                display=(
                    f"{federation.name:<{name_width}}        {federation.description}"
                ),
                selected_style="#ffffff bg:#dc8400 noreverse",
            )


def select_federation(prompt: str, federations: list[Federation]) -> str:
    """Return the federation selected by a command prompt."""
    federation_name = prompt[len(CHAT_FEDERATION_COMMAND) :].strip()
    if federation_name not in {federation.name for federation in federations}:
        raise click.ClickException(f"Unknown federation: {federation_name}")
    return federation_name


def resolve_default_federation(federations: list[Federation]) -> str | None:
    """Resolve the account-scoped default federation used by Flower Chat."""
    execution_suffix = f"/{CHAT_DEFAULT_FEDERATION_NAME}"
    for federation in federations:
        if federation.name.endswith(execution_suffix):
            return federation.name

    workspace_suffix = f"/{DEFAULT_FEDERATION_SIMULATION}"
    for federation in federations:
        if federation.name.endswith(workspace_suffix):
            account_name = federation.name[: -len(workspace_suffix)]
            return f"{account_name}{execution_suffix}"
    return None
