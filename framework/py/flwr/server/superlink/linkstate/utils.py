# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Utility functions for State."""


from collections.abc import Sequence
from uuid import UUID

from flwr.app import DEFAULT_TTL, ConfigRecord, Context, Message, Metadata, RecordDict
from flwr.app.message import make_message
from flwr.common.constant import SUPERLINK_NODE_ID
from flwr.proto.federation_config_pb2 import SimulationConfig  # pylint: disable=E0611
from flwr.supercore.constant import (
    AGENT_MESSAGE_CONTENT_RECORD_KEY,
    AGENT_MESSAGE_TEXT_KEY,
    SYSTEM_MESSAGE_TYPE,
    TaskType,
)
from flwr.supercore.corestate.utils import (
    generate_rand_int_from_bytes as corestate_generate_rand_int_from_bytes,
)
from flwr.supercore.date import now
from flwr.supercore.typing import AgentRunTarget
from flwr.supercore.utils import parse_app_spec


def validate_agent_app(target: AgentRunTarget, app_spec: str) -> None:
    """Require the accepted FAB to match the selected agent's configuration."""
    app_id, version = parse_app_spec(app_spec)
    if app_id != f"@{target.fab_id}" or (version and version != target.fab_version):
        raise ValueError("The resolved AgentApp does not match the selected agent.")


def validate_run_scope(  # pylint: disable=too-many-arguments
    *,
    federation_id: str | None,
    space_id: UUID | None,
    series_id: int | None,
    federation_config: SimulationConfig | None,
    series_description: str | None,
    connector_ids: Sequence[int],
    primary_task_type: str,
) -> None:
    """Require either a federation run or a Space AgentApp run."""
    if space_id is None:
        if not federation_id:
            raise ValueError("A run requires a federation_id or space_id.")
        return
    if federation_id is not None or series_id is not None:
        raise ValueError("space_id cannot be combined with federation_id or series_id.")
    if federation_config is not None or series_description is not None or connector_ids:
        raise ValueError("Space runs cannot use federation configuration or resources.")
    if primary_task_type != TaskType.AGENT_APP:
        raise ValueError("Space runs require an AgentApp primary task.")


def create_run_context(run_id: int, series_id: int = 0) -> Context:
    """Create an empty execution context for a run."""
    return Context(
        run_id=run_id,
        node_id=SUPERLINK_NODE_ID,
        node_config={},
        state=RecordDict(),
        run_config={},
        series_id=series_id,
    )


def create_user_prompt_message(run_id: int, user_prompt: str) -> Message:
    """Create a SuperLink instruction Message carrying an AgentApp user prompt."""
    metadata = Metadata(
        run_id=run_id,
        message_id="",
        src_node_id=SUPERLINK_NODE_ID,
        dst_node_id=SUPERLINK_NODE_ID,
        reply_to_message_id="",
        group_id="",
        message_type=SYSTEM_MESSAGE_TYPE,
        created_at=now().timestamp(),
        ttl=DEFAULT_TTL,
    )
    agent_record = ConfigRecord({AGENT_MESSAGE_TEXT_KEY: user_prompt})
    content = RecordDict({AGENT_MESSAGE_CONTENT_RECORD_KEY: agent_record})
    message = make_message(metadata=metadata, content=content)
    message.metadata.__dict__["_message_id"] = message.object_id
    return message


def generate_rand_int_from_bytes(
    num_bytes: int, exclude: set[int] | None = None
) -> int:
    """Generate a random unsigned integer from `num_bytes` bytes."""
    return corestate_generate_rand_int_from_bytes(num_bytes, exclude)
