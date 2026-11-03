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


from flwr.app import DEFAULT_TTL, ConfigRecord, Message, Metadata, RecordDict
from flwr.app.message import make_message
from flwr.common.constant import SUPERLINK_NODE_ID
from flwr.supercore.constant import (
    AGENT_MESSAGE_CONTENT_RECORD_KEY,
    AGENT_MESSAGE_TEXT_KEY,
    SYSTEM_MESSAGE_TYPE,
)
from flwr.supercore.corestate.utils import (
    generate_rand_int_from_bytes as corestate_generate_rand_int_from_bytes,
)
from flwr.supercore.date import now


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
