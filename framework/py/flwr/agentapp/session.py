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
from typing import Any, Protocol

from flwr.common import Context
from flwr.common.typing import Run
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    PushRunEventsRequest,
    RunEventPayload,
)
from flwr.supercore.task_message import JsonObject, JsonValue

from .conversation import AgentConversationClient
from .exceptions import AgentAppError
from .model import DEFAULT_AGENT_MODEL, AgentModelClient

AGENT_START_RECORD_KEY = "agent.start"
AGENT_REF_KEY = "agent_ref"
AGENT_INPUT_JSON_KEY = "input_json"
AGENT_CONVERSATION_ID_KEY = "conversation_id"
AGENT_MODEL_KEY = "model"


class ServerAppIoAgentSessionStub(Protocol):
    """Subset of ServerAppIo RPCs used by AgentSession."""

    def PushRunEvents(self, request: PushRunEventsRequest) -> Any:
        """Push run events."""


@dataclass(frozen=True)
class AgentStartState:
    """Validated AgentApp start state."""

    agent_ref: str
    input: JsonValue
    conversation_id: str
    model: str


class AgentSession:
    """Runtime session passed to AgentApp main functions."""

    def __init__(
        self,
        *,
        stub: Any,
        task_id: int,
        context: Context,
        run: Run,
        start_state: AgentStartState,
        model_response_timeout: float | None = None,
        model_poll_interval: float | None = None,
    ) -> None:
        if task_id <= 0:
            raise ValueError("`task_id` must be greater than zero.")
        self._stub = stub
        self.task_id = task_id
        self.context = context
        self.run = run
        self.agent_ref = start_state.agent_ref
        self.input = start_state.input
        self.conversation_id = start_state.conversation_id
        self.model = AgentModelClient(
            stub=stub,
            task_id=task_id,
            emit_event=self.emit_event,
            default_model=start_state.model,
            **(
                {"response_timeout": model_response_timeout}
                if model_response_timeout is not None
                else {}
            ),
            **(
                {"poll_interval": model_poll_interval}
                if model_poll_interval is not None
                else {}
            ),
        )
        self.conversation = AgentConversationClient(
            stub=stub,
            conversation_id=start_state.conversation_id,
        )

    @classmethod
    def from_context(
        cls,
        *,
        stub: Any,
        task_id: int,
        context: Context,
        run: Run,
        model_response_timeout: float | None = None,
        model_poll_interval: float | None = None,
    ) -> AgentSession:
        """Create an AgentSession from task input context."""
        return cls(
            stub=stub,
            task_id=task_id,
            context=context,
            run=run,
            start_state=parse_agent_start_state(context),
            model_response_timeout=model_response_timeout,
            model_poll_interval=model_poll_interval,
        )

    def emit_event(self, event: str, data: JsonObject) -> None:
        """Emit one compact JSON run event."""
        if not event:
            raise ValueError("`event` must be a non-empty string.")
        payload: JsonObject = {
            "task_id": self.task_id,
            "conversation_id": self.conversation_id,
        }
        payload.update(data)
        self._stub.PushRunEvents(
            PushRunEventsRequest(
                events=[
                    RunEventPayload(
                        event=event,
                        data=json.dumps(
                            payload,
                            separators=(",", ":"),
                            allow_nan=False,
                        ),
                    )
                ]
            )
        )


def parse_agent_start_state(context: Context) -> AgentStartState:
    """Parse AgentApp start state from a task context."""
    record = context.state.config_records.get(AGENT_START_RECORD_KEY)
    if record is None:
        raise AgentAppError("AgentApp context is missing `agent.start` state.")

    agent_ref = _required_str(record.get(AGENT_REF_KEY), AGENT_REF_KEY)
    input_json = _required_str(record.get(AGENT_INPUT_JSON_KEY), AGENT_INPUT_JSON_KEY)
    conversation_id = _required_str(
        record.get(AGENT_CONVERSATION_ID_KEY), AGENT_CONVERSATION_ID_KEY
    )
    model = record.get(AGENT_MODEL_KEY)
    model_ref = model if isinstance(model, str) and model else DEFAULT_AGENT_MODEL

    try:
        parsed_input = json.loads(input_json)
    except json.JSONDecodeError as exc:
        raise AgentAppError("AgentApp input_json must contain valid JSON.") from exc
    if not isinstance(parsed_input, (dict, list)):
        raise AgentAppError("AgentApp input_json must contain a JSON object or list.")

    return AgentStartState(
        agent_ref=agent_ref,
        input=parsed_input,
        conversation_id=conversation_id,
        model=model_ref,
    )


def _required_str(value: object, key: str) -> str:
    """Return a required string value."""
    if not isinstance(value, str) or not value:
        raise AgentAppError(f"AgentApp start state requires `{key}`.")
    return value
