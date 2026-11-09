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
"""Accept explicit agent triggers independently of message publication."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

from flwr.app.user_config import UserConfig
from flwr.proto.control_pb2 import SpaceMessage  # pylint: disable=E0611
from flwr.supercore.utils import parse_app_spec

if TYPE_CHECKING:
    from flwr.server.superlink.linkstate.linkstate import LinkState


@dataclass(frozen=True)
class AgentRunRequest:  # pylint: disable=too-many-instance-attributes
    """An explicit user-message target and its server-resolved AgentApp."""

    space_id: UUID
    agent_id: int
    message_id: int
    flwr_aid: str
    fab_id: str
    fab_version: str
    fab_hash: str
    override_config: UserConfig = field(default_factory=dict)


@dataclass(frozen=True)
class AgentRunTarget:
    """An explicit agent target with its already resolved AgentApp."""

    agent_id: int
    fab_id: str
    fab_version: str
    fab_hash: str
    override_config: UserConfig = field(default_factory=dict)

    def request_for(self, message: SpaceMessage, flwr_aid: str) -> AgentRunRequest:
        """Bind the target to the message accepted in this transaction."""
        return AgentRunRequest(
            space_id=UUID(message.space_id),
            agent_id=self.agent_id,
            message_id=message.id,
            flwr_aid=flwr_aid,
            fab_id=self.fab_id,
            fab_version=self.fab_version,
            fab_hash=self.fab_hash,
            override_config=self.override_config,
        )


def submit_agent_message(
    state: LinkState,
    space_id: UUID,
    flwr_aid: str,
    content: str,
    targets: Sequence[AgentRunTarget],
) -> tuple[SpaceMessage, list[int]]:
    """Atomically accept a message and pending executions for explicit targets.

    The caller resolves AgentApps and selects targets before submission. An empty
    target list publishes only the message. No participants are targeted implicitly.
    """
    if not flwr_aid or not content.strip():
        raise ValueError("An account and nonempty message content are required.")
    targets = tuple(targets)
    if len({target.agent_id for target in targets}) != len(targets):
        raise ValueError("Each agent may be targeted only once.")
    for target in targets:
        _validate_target(target)
    return state.submit_agent_message(space_id, flwr_aid, content, targets)


def enqueue_agent_run(state: LinkState, request: AgentRunRequest) -> int:
    """Accept one target, returning the original Run on redelivery.

    For new messages, use submit_agent_message to avoid a gap before enqueueing.
    Late assignments are rejected if a newer message's execution has started.
    """
    if not request.flwr_aid or not 0 < request.message_id < 2**63:
        raise ValueError(
            "An account and positive signed 64-bit message ID are required."
        )
    _validate_target(request)
    return state.enqueue_agent_run(request)


def _validate_target(target: AgentRunTarget | AgentRunRequest) -> None:
    """Require a persisted agent identity and resolved FAB coordinates."""
    if not 0 < target.agent_id < 2**63:
        raise ValueError("Agent IDs must be positive signed 64-bit integers.")
    if not all((target.fab_id, target.fab_version, target.fab_hash)):
        raise ValueError("A resolved AgentApp is required.")


def validate_agent_app(request: AgentRunRequest, app_spec: str) -> None:
    """Require the accepted FAB to match the selected agent's configuration."""
    app_id, version = parse_app_spec(app_spec)
    if app_id != f"@{request.fab_id}" or (version and version != request.fab_version):
        raise ValueError("The resolved AgentApp does not match the selected agent.")
