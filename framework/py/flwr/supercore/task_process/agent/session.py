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
import time
from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from logging import ERROR
from queue import Empty, Queue
from threading import Lock, Thread
from typing import cast
from uuid import uuid4

from google.protobuf.json_format import ParseDict
from httpx import HTTPStatusError

from flwr.agentapp import AgentConnectors, AgentEvents, AgentGrid, AgentSession
from flwr.app import Message
from flwr.common.logger import log
from flwr.common.serde import message_from_proto, message_to_proto

# pylint: disable=E0611
from flwr.proto.control_pb2 import StartAutomationRequest, StartRunRequest
from flwr.proto.runseries_pb2 import CompactedContext
from flwr.proto.runtime_pb2 import (
    CreateTaskRequest,
    GetCompactedContextsRequest,
    GetRunSeriesEventsRequest,
    PullTaskMessageRequest,
    PushTaskEventsRequest,
    PushTaskMessageRequest,
    PutCompactedContextRequest,
)
from flwr.proto.task_pb2 import TaskEvent

# pylint: enable=E0611
from flwr.supercore.constant import TaskType
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.json_message.connector_message import (
    ConnectorRequest,
    ConnectorResponse,
)
from flwr.supercore.runtime import RuntimeHttpClient
from flwr.supercore.task_process.connector.automation import START_AUTOMATION_TOOL_NAME
from flwr.supercore.task_process.connector.registry import (
    get_connector_discovery_tool,
    get_connector_ref,
    get_connector_tools,
    is_dynamic_connector_tool,
)
from flwr.supercore.typing import JSONObject, JSONValue
from flwr.supercore.utils import strict_json_dumps, strict_json_loads

_DEFAULT_TASK_REPLY_TIMEOUT = 300.0
_DEFAULT_TASK_REPLY_POLL_INTERVAL = 0.25
_EVENT_PUBLISH_BATCH_SIZE = 16
_EVENT_PUBLISH_QUEUE_SIZE = 256
_EVENT_PUBLISH_BATCH_WAIT = 0.05
_EVENT_PUBLISH_STOP = object()


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

    def get_trace(
        self,
        *,
        event_types: Sequence[str] | None = None,
        after_event_id: int | None = None,
        before_event_id: int | None = None,
        last_n: int | None = None,
    ) -> list[JSONObject]:
        """Get matching events from the current run series."""
        request = GetRunSeriesEventsRequest(
            event_types=event_types or [],
            last_n=0 if last_n is None else last_n,
            after_event_id=after_event_id,
            before_event_id=before_event_id,
        )
        response = self._stub.GetRunSeriesEvents(request)
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

    def get_compacted_contexts(
        self, *, limit: int = 1, before_context_id: int | None = None
    ) -> list[JSONObject]:
        """Get summaries from the current run series."""
        response = self._stub.GetCompactedContexts(
            GetCompactedContextsRequest(
                limit=limit, before_context_id=before_context_id
            )
        )
        return [
            {
                "id": context.id,
                "series_id": context.series_id,
                "through_event_id": context.through_event_id,
                "text": context.text,
                "created_at": context.created_at,
                "token_count": context.token_count,
            }
            for context in response.contexts
        ]

    def put_compacted_context(
        self, *, through_event_id: int, text: str, token_count: int
    ) -> None:
        """Store a summary in the current run series."""
        self._stub.PutCompactedContext(
            PutCompactedContextRequest(
                context=CompactedContext(
                    through_event_id=through_event_id,
                    text=text,
                    token_count=token_count,
                )
            )
        )

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
        prompt: str,
        connectors: AgentConnectors,
        events: AgentEvents,
        grid: AgentGrid,
    ) -> None:
        self._prompt = prompt
        self._connectors = connectors
        self._events = events
        self._grid = grid

    @property
    def prompt(self) -> str:
        """Return the initial prompt for this AgentApp run."""
        return self._prompt

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


class RuntimeAgentConnectors(AgentConnectors):
    """AgentConnectors implementation for model tools."""

    def __init__(self, agent_runtime: AgentRuntime) -> None:
        self._agent_runtime = agent_runtime

    def tools(self, names: Sequence[str], *, refresh: bool = False) -> list[JSONObject]:
        """Return schemas, optionally refreshing the runtime discovery cache."""
        return [
            tool
            for name in names
            for tool in self._agent_runtime.get_connector_tools(name, refresh=refresh)
        ]

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


# pylint: disable-next=too-many-instance-attributes
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
        self._discovered_tools: dict[str, list[JSONObject]] = {}
        self._discovery_locks: defaultdict[str, Lock] = defaultdict(Lock)
        self._discovery_lock = Lock()

    def get_connector_tools(
        self, connector_ref: str, *, refresh: bool = False
    ) -> list[JSONObject]:
        """Return static tools plus a copied, run-scoped dynamic catalog."""
        tools = get_connector_tools(connector_ref)
        if get_connector_discovery_tool(connector_ref) is None:
            return tools
        try:
            discovered = deepcopy(
                self._discover_connector_tools(connector_ref, refresh=refresh)
            )
        except HTTPStatusError as exc:
            error = FlowerError.from_json(exc.response.text)
            if (
                exc.response.status_code == 403
                and error is not None
                and error.code == ApiErrorCode.RUNTIME_CONNECTOR_NOT_AVAILABLE
            ):
                log(
                    ERROR,
                    "Skipping tool discovery for unavailable connector %s "
                    "in run %d, task %d (API error %d).",
                    connector_ref,
                    self._run_id,
                    self._task_id,
                    error.code,
                )
                return []
            raise
        for tool in discovered:
            tool["name"] = cast(str, tool["name"]).lower()
        return tools + discovered

    def _discover_connector_tools(
        self, connector_ref: str, *, refresh: bool = False
    ) -> list[JSONObject]:
        """Cache tool metadata and serialize discovery only within each connector."""
        with self._discovery_lock:
            lock = self._discovery_locks[connector_ref]
        with lock:
            if refresh:
                self._discovered_tools.pop(connector_ref, None)
            if connector_ref in self._discovered_tools:
                return self._discovered_tools[connector_ref]
            operation = get_connector_discovery_tool(connector_ref)
            if operation is None:
                raise ValueError(
                    f"Connector '{connector_ref}' has no discovery source."
                )
            result = self.create_connector_response(
                name=operation, call_id=uuid4().hex, arguments={}
            )
            if not isinstance(result, list) or not all(
                isinstance(tool, dict) for tool in result
            ):
                raise RuntimeError(
                    f"Connector '{connector_ref}' returned invalid tools."
                )
            tools = cast(list[JSONObject], result)
            names: set[str] = {operation}
            names.update(
                cast(str, tool["name"]).lower()
                for tool in get_connector_tools(connector_ref)
            )
            for tool in tools:
                name = tool.get("name")
                if (
                    not isinstance(name, str)
                    or not name.startswith(f"{connector_ref}_")
                    or name == f"{connector_ref}_"
                    or name.lower() in names
                    or get_connector_ref(name.lower()) != connector_ref
                ):
                    raise RuntimeError(
                        f"Connector '{connector_ref}' returned invalid tool names."
                    )
                names.add(name.lower())
            self._discovered_tools[connector_ref] = tools
            return tools

    def create_connector_response(
        self, *, name: str, call_id: str, arguments: JSONObject
    ) -> JSONValue:
        """Create one connector response."""
        name = name.strip().lower()
        connector_ref = get_connector_ref(name)
        if is_dynamic_connector_tool(name):
            catalog = self._discover_connector_tools(connector_ref)
            tool = next(
                (tool for tool in catalog if cast(str, tool["name"]).lower() == name),
                None,
            )
            if tool is None:
                raise ValueError(f"Unsupported discovered tool '{name}'.")
            name = cast(str, tool["name"])
        create_res = self._stub.CreateTask(
            CreateTaskRequest(type=TaskType.CONNECTOR, connector_ref=connector_ref)
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
            request.start_run_request.user_prompt = input_value.strip()
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
