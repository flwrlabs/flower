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
"""Executor-bound AgentApp session implementations."""


from __future__ import annotations

import json
import random
import time
from collections.abc import Sequence
from queue import Empty, Queue
from threading import Lock, Thread
from typing import cast

from google.protobuf.json_format import MessageToDict, ParseDict

from flwr.agentapp import AgentConnectors, AgentEvents, AgentGrid, AgentSession
from flwr.agentapp.constants import (
    AGENT_GRID_MESSAGE_PAYLOAD_JSON_KEY,
    AGENT_GRID_MESSAGE_PAYLOAD_RECORD_KEY,
)
from flwr.app import ConfigRecord, Message, RecordDict
from flwr.common.serde import message_from_proto, message_to_proto

# pylint: disable=E0611
from flwr.proto.control_pb2 import StartAutomationRequest, StartRunRequest
from flwr.proto.runtime_pb2 import (
    CreateTaskRequest,
    GetRunSeriesEventsRequest,
    PullTaskMessageRequest,
    PushTaskEventsRequest,
    PushTaskMessageRequest,
)
from flwr.proto.task_pb2 import TaskEvent

# pylint: enable=E0611
from flwr.serverapp import Grid
from flwr.supercore.constant import TaskType
from flwr.supercore.json_message.connector_message import (
    ConnectorRequest,
    ConnectorResponse,
)
from flwr.supercore.runtime import RuntimeHttpClient
from flwr.supercore.task_process.connector.automation import START_AUTOMATION_TOOL_NAME
from flwr.supercore.task_process.connector.registry import (
    get_connector_ref,
    get_connector_tools,
)
from flwr.supercore.task_process.connector.tool_schema import (
    function_tool,
    string_property,
)
from flwr.supercore.typing import JSONObject, JSONValue
from flwr.supercore.utils import strict_json_dumps, strict_json_loads

_DEFAULT_TASK_REPLY_TIMEOUT = 300.0
_DEFAULT_TASK_REPLY_POLL_INTERVAL = 0.25
_EVENT_PUBLISH_BATCH_SIZE = 16
_EVENT_PUBLISH_QUEUE_SIZE = 256
_EVENT_PUBLISH_BATCH_WAIT = 0.05
_EVENT_PUBLISH_STOP = object()
_GRID_TOOL_NAMES = {"get_nodes", "push_messages", "pull_messages"}


def _grid_tools() -> list[JSONObject]:
    """Return model-facing federation Grid tool schemas."""
    return [
        function_tool(
            "get_nodes",
            "Return all available SuperNodes, or a random sample if requested.",
            properties={
                "sample_size": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional maximum number of SuperNodes to return.",
                }
            },
        ),
        function_tool(
            "push_messages",
            (
                "Send messages to SuperNodes and return one result per message in "
                "the same order. Pass accepted message IDs to pull_messages if "
                "replies are required."
            ),
            properties={
                "messages": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "dst_node_id": string_property(
                                "Destination SuperNode ID as a decimal string."
                            ),
                            "payload": {
                                "type": "object",
                                "description": "JSON object to send.",
                            },
                            "ttl": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                                "description": "Optional round-trip TTL in seconds.",
                            },
                        },
                        "required": ["dst_node_id", "payload"],
                        "additionalProperties": False,
                    },
                },
            },
            required=["messages"],
        ),
        function_tool(
            "pull_messages",
            "Wait for replies to message IDs returned by push_messages.",
            properties={
                "message_ids": {
                    "type": "array",
                    "items": string_property("Message ID returned by push_messages."),
                    "minItems": 1,
                    "description": "Message IDs whose replies are awaited.",
                },
                "timeout": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 300,
                    "description": "Maximum wait in seconds; zero checks once.",
                },
            },
            required=["message_ids", "timeout"],
        ),
    ]


class RuntimeAgentEvents(AgentEvents):
    """Publish AgentApp-selected events through a background worker."""

    def __init__(self, stub: RuntimeHttpClient) -> None:
        self._stub = stub
        self._queue: Queue[TaskEvent | object] = Queue(
            maxsize=_EVENT_PUBLISH_QUEUE_SIZE
        )
        self._error_lock = Lock()
        self._error: Exception | None = None
        self._closed = False
        self._worker = Thread(
            target=self._run,
            name="flwr-agent-event-publisher",
            daemon=True,
        )
        self._worker.start()

    def get_trace(self) -> list[JSONObject]:
        """Get events from all runs in the current run series."""
        response = self._stub.GetRunSeriesEvents(GetRunSeriesEventsRequest())
        return [
            {
                "id": event.id,
                "timestamp": event.timestamp,
                "run_id": event.run_id,
                "task_id": event.task_id,
                "event": event.event,
                "data": strict_json_loads(event.data),
            }
            for event in response.events
        ]

    def emit(self, event: JSONObject) -> None:
        """Queue one event for publication to run-event subscribers."""
        if self._closed:
            raise RuntimeError("Agent event publisher is closed.")
        event_type = event.get("type")
        if not isinstance(event_type, str) or not event_type:
            raise ValueError("Run event requires a non-empty string 'type' field.")
        self._raise_worker_error()
        task_event = TaskEvent(
            event=event_type,
            data=strict_json_dumps(event, compact=True),
        )
        self._queue.put(task_event)
        self._raise_worker_error()

    def close(self, timeout: float | None = None) -> None:
        """Publish pending events and stop the background publisher."""
        if self._closed:
            self._raise_worker_error()
            return

        self._closed = True
        self._queue.put(_EVENT_PUBLISH_STOP)
        self._worker.join(timeout)
        if self._worker.is_alive():
            raise TimeoutError("Timed out waiting for Agent event publisher to stop.")
        self._raise_worker_error()

    def _flush(self, batch: list[TaskEvent]) -> None:
        """Publish one batch of task events."""
        try:
            self._stub.PushTaskEvents(PushTaskEventsRequest(events=batch))
        except Exception as err:  # pylint: disable=broad-exception-caught
            with self._error_lock:
                if self._error is None:
                    self._error = err

    def _run(self) -> None:
        """Upload queued events in small batches."""
        while True:
            item = self._queue.get()
            if item is _EVENT_PUBLISH_STOP:
                return

            batch = [cast(TaskEvent, item)]
            deadline = time.monotonic() + _EVENT_PUBLISH_BATCH_WAIT
            while len(batch) < _EVENT_PUBLISH_BATCH_SIZE:
                try:
                    item = self._queue.get(
                        timeout=max(0.0, deadline - time.monotonic())
                    )
                except Empty:
                    break

                if item is _EVENT_PUBLISH_STOP:
                    self._flush(batch)
                    return

                batch.append(cast(TaskEvent, item))

            self._flush(batch)

    def _raise_worker_error(self) -> None:
        """Raise a background publication failure in the AgentApp thread."""
        with self._error_lock:
            error = self._error
        if error is not None:
            raise RuntimeError("Failed to publish AgentApp events.") from error


class RuntimeAgentSession(AgentSession):
    """AgentSession bound to one AgentApp task."""

    def __init__(
        self,
        connectors: AgentConnectors,
        events: AgentEvents,
        grid: AgentGrid,
    ) -> None:
        self._connectors = connectors
        self._events = events
        self._grid = grid

    @property
    def connectors(self) -> AgentConnectors:
        """Connector tool schema and execution API."""
        return self._connectors

    @property
    def events(self) -> AgentEvents:
        """Frontend-visible structured run event API."""
        return self._events

    @property
    def grid(self) -> AgentGrid:
        """Model-facing federation Grid API."""
        return self._grid


class RuntimeAgentGrid(AgentGrid):
    """Expose selected Grid operations as model tools."""

    def __init__(self, grid: Grid, events: AgentEvents) -> None:
        self._grid = grid
        self._events = events

    def tools(self) -> list[JSONObject]:
        """Return model-facing Grid tool schemas."""
        return _grid_tools()

    def call(self, tool_call: JSONObject) -> JSONObject:
        """Execute one Grid function_call and return a function_call_output item."""
        arguments = tool_call["arguments"]
        if isinstance(arguments, str):
            arguments = strict_json_loads(arguments)
        name = cast(str, tool_call["name"])
        call_id = cast(str, tool_call["call_id"])
        if name not in _GRID_TOOL_NAMES:
            raise ValueError(f"Unsupported Grid tool '{name}'.")

        arguments_obj = cast(JSONObject, arguments)
        self._events.emit(
            {
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": strict_json_dumps(arguments_obj, compact=True),
            }
        )
        output = cast(JSONObject, getattr(self, f"_{name}")(**arguments_obj))
        output_item: JSONObject = {
            "type": "function_call_output",
            "call_id": call_id,
            "output": strict_json_dumps(output, compact=True),
        }
        self._events.emit(output_item)
        return output_item

    def _get_nodes(self, sample_size: int | None = None) -> JSONObject:
        node_ids = list(self._grid.get_node_ids())
        if sample_size is not None and sample_size < 1:
            raise ValueError("Grid sample size must be positive.")
        selected = (
            node_ids
            if sample_size is None
            else random.sample(node_ids, min(sample_size, len(node_ids)))
        )
        return {
            "node_ids": [str(node_id) for node_id in selected],
            "num_available": len(node_ids),
        }

    def _push_messages(self, messages: list[JSONObject]) -> JSONObject:
        if not messages:
            raise ValueError("At least one message is required.")

        outgoing = []
        for item in messages:
            payload = cast(JSONObject, item["payload"])
            outgoing.append(
                Message(
                    RecordDict(
                        {
                            AGENT_GRID_MESSAGE_PAYLOAD_RECORD_KEY: ConfigRecord(
                                {
                                    AGENT_GRID_MESSAGE_PAYLOAD_JSON_KEY: (
                                        strict_json_dumps(payload, compact=True)
                                    )
                                }
                            )
                        }
                    ),
                    dst_node_id=int(cast(str, item["dst_node_id"])),
                    message_type="query",  # Replace with an AgentGrid message type.
                    group_id="",
                    ttl=cast(float | None, item.get("ttl")),
                )
            )

        message_ids = list(self._grid.push_messages(outgoing))
        if len(message_ids) != len(outgoing):
            raise RuntimeError("Grid returned an unexpected number of message IDs.")
        return {
            "results": [
                {
                    "message_id": message_id or None,
                    "error": None if message_id else "Message was not accepted.",
                }
                for message_id in message_ids
            ]
        }

    def _pull_messages(self, message_ids: list[str], timeout: float) -> JSONObject:
        if not 0 <= timeout <= 300:
            raise ValueError("Grid pull timeout must be between 0 and 300 seconds.")
        pending = set(message_ids)
        replies: list[Message] = []
        deadline = time.monotonic() + timeout
        while pending:
            pulled = list(self._grid.pull_messages(pending))
            replies.extend(pulled)
            pending.difference_update(
                message.metadata.reply_to_message_id for message in pulled
            )
            remaining = deadline - time.monotonic()
            if not pending or remaining <= 0:
                break
            time.sleep(min(0.25, remaining))
        return {
            "messages": [
                cast(JSONObject, MessageToDict(message_to_proto(message)))
                for message in replies
            ],
            "pending_message_ids": sorted(pending),
        }


class RuntimeAgentConnectors(AgentConnectors):
    """AgentConnectors implementation for model tools."""

    def __init__(self, agent_runtime: AgentRuntime) -> None:
        self._agent_runtime = agent_runtime

    def tools(self, names: Sequence[str]) -> list[JSONObject]:
        """Return model-facing tool schemas for the requested connectors."""
        return [tool for name in names for tool in get_connector_tools(name)]

    def call(self, tool_call: JSONObject) -> JSONObject:
        """Execute one model function_call and return a function_call_output item."""
        arguments = tool_call["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        name = cast(str, tool_call["name"])
        call_id = cast(str, tool_call["call_id"])
        arguments_obj = cast(JSONObject, arguments)

        if name == START_AUTOMATION_TOOL_NAME:
            return self._agent_runtime.call_automation_with_events(
                call_id=call_id,
                arguments=arguments_obj,
            )
        return self._agent_runtime.call_connector_with_events(
            name=name,
            call_id=call_id,
            arguments=arguments_obj,
        )


class AgentRuntime:
    """Coordinate AgentApp operations with Runtime services."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        stub: RuntimeHttpClient,
        run_id: int,
        task_id: int,
        start_run_request: StartRunRequest,
        events: AgentEvents,
    ) -> None:
        self._stub = stub
        self._run_id = run_id
        self._task_id = task_id
        self._start_run_request = start_run_request
        self._events = events

    def create_connector_response(
        self, *, name: str, call_id: str, arguments: JSONObject
    ) -> JSONValue:
        """Create one connector response through a child connector task."""
        name = name.strip().lower()
        create_res = self._stub.CreateTask(
            CreateTaskRequest(
                type=TaskType.CONNECTOR, connector_ref=get_connector_ref(name)
            )
        )
        if not create_res.HasField("task_id"):
            raise RuntimeError("Connector task could not be created.")

        connector_task_id = create_res.task_id
        message = ConnectorRequest(
            dst_task_id=connector_task_id,
            name=name,
            call_id=call_id,
            arguments=arguments,
        )
        response_message = self._send_and_receive(message)
        response = ConnectorResponse.from_message(response_message)
        response_payload = response.payload

        error = response_payload.get("error")
        if error is not None:
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                raise RuntimeError(f"Connector '{name}' failed: {error['message']}")
            raise RuntimeError(f"Connector '{name}' failed.")

        return response_payload["output"]

    def call_connector_with_events(
        self, *, name: str, call_id: str, arguments: JSONObject
    ) -> JSONObject:
        """Call a connector and emit/persist its activity events."""
        name = name.strip().lower()
        function_call: JSONObject = {
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": strict_json_dumps(arguments, compact=True),
        }
        self.push_run_events([function_call])

        try:
            output = self.create_connector_response(
                name=name,
                call_id=call_id,
                arguments=arguments,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            error_output: JSONObject = {
                "error": {
                    "code": "connector_error",
                    "message": "Connector execution failed.",
                }
            }
            self.push_run_events(
                [
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": strict_json_dumps(error_output, compact=True),
                    }
                ]
            )
            raise

        output_item: JSONObject = {
            "type": "function_call_output",
            "call_id": call_id,
            "output": strict_json_dumps(output, compact=True),
        }
        self.push_run_events([output_item])
        return output_item

    def call_automation_with_events(
        self, *, call_id: str, arguments: JSONObject
    ) -> JSONObject:
        """Create an automation and emit/persist its activity events."""
        function_call: JSONObject = {
            "type": "function_call",
            "call_id": call_id,
            "name": START_AUTOMATION_TOOL_NAME,
            "arguments": strict_json_dumps(arguments, compact=True),
        }
        self.push_run_events([function_call])
        try:
            input_value = arguments.get("input")
            if not isinstance(input_value, str) or not input_value.strip():
                raise ValueError("Automation input must be a non-empty string.")
            start_at = arguments.get("start_at")
            if not isinstance(start_at, str) or not start_at.strip():
                raise ValueError("Automation start_at must be a non-empty string.")
            request_data = dict(arguments)
            del request_data["input"]
            request = ParseDict(
                request_data,
                StartAutomationRequest(
                    start_run_request=self._start_run_request,
                ),
            )
            request.start_run_request.override_config["agent.input"].string = (
                input_value.strip()
            )
            response = self._stub.StartAutomation(request)
            output: JSONObject = {
                "automation_id": response.automation_id,
                "series_id": response.series_id,
                "next_run_at": response.next_run_at,
            }
        except Exception:  # pylint: disable=broad-exception-caught
            error_output: JSONObject = {
                "error": {
                    "code": "automation_error",
                    "message": "Automation execution failed.",
                }
            }
            self.push_run_events(
                [
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": strict_json_dumps(error_output, compact=True),
                    }
                ]
            )
            raise

        output_item: JSONObject = {
            "type": "function_call_output",
            "call_id": call_id,
            "output": strict_json_dumps(output, compact=True),
        }
        self.push_run_events([output_item])
        return output_item

    def push_run_events(self, events: Sequence[JSONObject]) -> None:
        """Queue structured run events for `StreamRunEvents` clients."""
        for event in events:
            self._events.emit(event)

    def _push_task_message(self, message: Message) -> None:
        """Push one task message and return its message ID."""
        message.metadata.__dict__["_run_id"] = self._run_id
        message.metadata.src_task_id = self._task_id
        message.metadata.__dict__["_message_id"] = message.object_id
        self._stub.PushTaskMessage(
            PushTaskMessageRequest(message=message_to_proto(message))
        )

    def _pull_task_messages(self, src_task_id: int) -> list[Message]:
        """Pull pending task messages from one child task."""
        res = self._stub.PullTaskMessage(
            PullTaskMessageRequest(limit=1, src_task_id=src_task_id)
        )
        return [message_from_proto(msg) for msg in res.messages]

    def _send_and_receive(self, message: Message) -> Message:
        """Send one message and wait for its destination child task's direct reply.

        For now, `flwr-agentapp` expects a strict one-request-one-reply exchange with
        child tasks, so any non-matching pulled message is treated as an error.
        """
        child_task_id = message.metadata.dst_task_id
        if child_task_id is None:
            raise ValueError("Task message requires a destination task ID.")

        # Push the message to the child task
        self._push_task_message(message)
        message_id = message.metadata.message_id

        # Pull until a message arrives that replies to the pushed message, or timeout
        deadline = time.monotonic() + _DEFAULT_TASK_REPLY_TIMEOUT
        while True:
            # The request destination becomes the source of its reply.
            for pulled_msg in self._pull_task_messages(src_task_id=child_task_id):
                if pulled_msg.metadata.reply_to_message_id != message_id:
                    raise RuntimeError(
                        "Received a message that does not reply to the request."
                    )
                return pulled_msg

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for child task response.")

            time.sleep(min(_DEFAULT_TASK_REPLY_POLL_INTERVAL, remaining))
