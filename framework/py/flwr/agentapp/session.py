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
"""AgentApp session."""


from __future__ import annotations

import json
from dataclasses import dataclass
from typing import cast

from flwr.common import Context
from flwr.common.typing import Run

from .exceptions import AgentAppError
from .typing import JSONObject, JSONValue

_AGENT_START_RECORD_KEY = "agent.start"
_AGENT_REF_KEY = "agent_ref"
_AGENT_INPUT_JSON_KEY = "input_json"
_AGENT_CONVERSATION_ID_KEY = "conversation_id"


@dataclass(frozen=True)
class _AgentStartState:
    """Validated AgentApp start state."""

    agent_ref: str
    input_items: list[JSONObject]
    conversation_id: str


class AgentSession:
    """Runtime session passed to AgentApp main functions."""

    def __init__(
        self,
        *,
        task_id: int,
        run: Run,
        agent_ref: str,
        conversation_id: str,
        input_items: list[JSONObject],
    ) -> None:
        if task_id <= 0:
            raise ValueError("`task_id` must be greater than zero.")
        if not agent_ref:
            raise ValueError("`agent_ref` must be a non-empty string.")
        if not conversation_id:
            raise ValueError("`conversation_id` must be a non-empty string.")
        if not all(isinstance(item, dict) for item in input_items):
            raise ValueError("`input_items` must be a list of JSON objects.")
        self.task_id = task_id
        self.run = run
        self.agent_ref = agent_ref
        self.conversation_id = conversation_id
        self.input_items = input_items

    @classmethod
    def from_context(
        cls,
        *,
        task_id: int,
        run: Run,
        context: Context,
    ) -> AgentSession:
        """Create an AgentSession from task input context."""
        start_state = _parse_agent_start_state(context)
        return cls(
            task_id=task_id,
            run=run,
            agent_ref=start_state.agent_ref,
            conversation_id=start_state.conversation_id,
            input_items=start_state.input_items,
        )


def _parse_agent_start_state(context: Context) -> _AgentStartState:
    """Parse AgentApp start state from a task context."""
    record = context.state.config_records.get(_AGENT_START_RECORD_KEY)
    if record is None:
        raise AgentAppError("AgentApp context is missing `agent.start` state.")

    agent_ref = _required_str(record.get(_AGENT_REF_KEY), _AGENT_REF_KEY)
    input_json = _required_str(record.get(_AGENT_INPUT_JSON_KEY), _AGENT_INPUT_JSON_KEY)
    conversation_id = _required_str(
        record.get(_AGENT_CONVERSATION_ID_KEY), _AGENT_CONVERSATION_ID_KEY
    )
    try:
        parsed_input = json.loads(input_json)
    except json.JSONDecodeError as exc:
        raise AgentAppError("AgentApp input_json must contain valid JSON.") from exc

    return _AgentStartState(
        agent_ref=agent_ref,
        input_items=_input_items_from_json_value(cast(JSONValue, parsed_input)),
        conversation_id=conversation_id,
    )


def _input_items_from_json_value(value: JSONValue) -> list[JSONObject]:
    """Return parsed input as a list of JSON objects."""
    if isinstance(value, dict):
        return [cast(JSONObject, value)]
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return [cast(JSONObject, item) for item in value]
    raise AgentAppError(
        "AgentApp input_json must contain a JSON object or list of JSON objects."
    )


def _required_str(value: object, key: str) -> str:
    """Return a required string value."""
    if not isinstance(value, str) or not value:
        raise AgentAppError(f"AgentApp start state requires `{key}`.")
    return value
