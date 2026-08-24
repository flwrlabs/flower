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
from collections.abc import Sequence
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Literal, cast

from google.protobuf.json_format import ParseDict

from flwr.agentapp import AgentConnectors, AgentEvents, AgentResponses, AgentSession
from flwr.app import Context, Message
from flwr.common.serde import message_from_proto, message_to_proto
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    StartAutomationRequest,
    StartRunRequest,
)
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    CreateTaskRequest,
    PullTaskMessageRequest,
    PushTaskEventsRequest,
    PushTaskMessageRequest,
)
from flwr.proto.task_pb2 import TaskEvent  # pylint: disable=E0611
from flwr.supercore.constant import TaskType
from flwr.supercore.json_message.connector_message import (
    ConnectorRequest,
    ConnectorResponse,
)
from flwr.supercore.json_message.model_message import ModelRequest, ModelResponse
from flwr.supercore.runtime import RuntimeHttpClient
from flwr.supercore.task_process.connector.automation import START_AUTOMATION_TOOL_NAME
from flwr.supercore.task_process.connector.registry import (
    get_connector_ref,
    get_connector_tools,
    has_builtin_connector,
)
from flwr.supercore.task_process.connector.web_fetch import WEB_FETCH_CONNECTOR_NAME
from flwr.supercore.typing import JSONObject, JSONValue
from flwr.supercore.utils import strict_json_dumps

from .context_items import append_items

_DEFAULT_MODEL_REPLY_TIMEOUT = 300.0
_DEFAULT_MODEL_REPLY_POLL_INTERVAL = 0.25
_EVENT_PUBLISH_BATCH_SIZE = 16
_EVENT_PUBLISH_QUEUE_SIZE = 256
_EVENT_PUBLISH_BATCH_WAIT = 0.002
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
        self._queue.put_nowait(_EVENT_PUBLISH_STOP)
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
        responses: AgentResponses,
        connectors: AgentConnectors,
        events: AgentEvents,
    ) -> None:
        self._responses = responses
        self._connectors = connectors
        self._events = events

    @property
    def responses(self) -> AgentResponses:
        """Model response creation API."""
        return self._responses

    @property
    def connectors(self) -> AgentConnectors:
        """Connector tool schema and execution API."""
        return self._connectors

    @property
    def events(self) -> AgentEvents:
        """Frontend-visible structured run event API."""
        return self._events


class RuntimeAgentConnectors(AgentConnectors):
    """AgentConnectors implementation for model tools."""

    def __init__(self, responses: RuntimeAgentResponses) -> None:
        self._responses = responses

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
            return self._responses.call_automation_with_events(
                call_id=call_id,
                arguments=arguments_obj,
            )
        return self._responses.call_connector_with_events(
            name=name,
            call_id=call_id,
            arguments=arguments_obj,
        )


class RuntimeAgentResponses(AgentResponses):
    """AgentResponses implementation backed by Runtime task messages."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        stub: RuntimeHttpClient,
        run_id: int,
        task_id: int,
        context: Context,
        start_run_request: StartRunRequest,
        events: AgentEvents,
    ) -> None:
        self._stub = stub
        self._context = context
        self._run_id = run_id
        self._task_id = task_id
        self._start_run_request = start_run_request
        self._events = events

    def create(self, request: JSONObject) -> JSONObject:
        """Create a model response through a child model task."""
        response_payload = self._create_model_response(request)

        output = response_payload.get("output")
        if _is_json_object_list(output):
            append_items(self._context, cast(list[JSONObject], output))
        return response_payload

    def _create_model_response(self, request: JSONObject) -> JSONObject:
        """Create one model response through a child model task."""
        model = request.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError(
                "AgentResponses request requires a non-empty string 'model' field."
            )

        create_res = self._stub.CreateTask(
            CreateTaskRequest(type=TaskType.MODEL, model_ref=model)
        )
        if not create_res.HasField("task_id"):
            raise RuntimeError("Model task could not be created.")

        model_task_id = create_res.task_id
        message = ModelRequest(
            dst_task_id=model_task_id,
            input_=cast(str | Sequence[JSONObject], request.get("input")),
            model=model,
            stream=cast(bool, request.get("stream", False)),
            tools=cast(Sequence[JSONObject] | None, request.get("tools")),
            tool_choice=request.get("tool_choice"),
            reasoning=cast(JSONObject | None, request.get("reasoning")),
            previous_response_id=cast(str | None, request.get("previous_response_id")),
            instructions=cast(str | None, request.get("instructions")),
            max_output_tokens=cast(int | None, request.get("max_output_tokens")),
            metadata=cast(JSONObject | None, request.get("metadata")),
            text=cast(JSONObject | None, request.get("text")),
        )
        response_message = self._send_and_receive(message)
        response = ModelResponse.from_message(response_message)
        return response.payload

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

        def connector_event(
            status: Literal["started", "completed", "failed"],
            *,
            output: JSONValue = None,
            message: str | None = None,
        ) -> list[JSONObject]:
            if not has_builtin_connector(name):
                return []

            event: JSONObject = {
                "type": f"response.tool_call.{status}",
                "tool_call_id": call_id,
                "connector_ref": name,
                "arguments": arguments,
            }

            query = arguments.get("query")
            if isinstance(query, str) and query:
                event["query"] = query

            url = arguments.get("url")
            if name == WEB_FETCH_CONNECTOR_NAME and isinstance(url, str) and url:
                event["links"] = [url]

            if status == "completed":
                event["output"] = output
            elif status == "failed" and message is not None:
                event["error"] = {"code": "connector_error", "message": message}

            return [event]

        self.append_and_push_run_events(connector_event("started"))

        try:
            output = self.create_connector_response(
                name=name,
                call_id=call_id,
                arguments=arguments,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.append_and_push_run_events(connector_event("failed", message=str(exc)))
            raise

        output_item: JSONObject = {
            "type": "function_call_output",
            "call_id": call_id,
            "output": strict_json_dumps(output, compact=True),
        }
        self.append_and_push_run_events(connector_event("completed", output=output))
        self.append_context_items([output_item])
        return output_item

    def call_automation_with_events(
        self, *, call_id: str, arguments: JSONObject
    ) -> JSONObject:
        """Create an automation and emit/persist its activity events."""

        def automation_event(
            status: Literal["started", "completed", "failed"],
            *,
            output: JSONValue = None,
            message: str | None = None,
        ) -> JSONObject:
            event: JSONObject = {
                "type": f"response.tool_call.{status}",
                "tool_call_id": call_id,
                "tool_name": START_AUTOMATION_TOOL_NAME,
                "arguments": arguments,
            }
            if status == "completed":
                event["output"] = output
            elif status == "failed" and message is not None:
                event["error"] = {
                    "code": "automation_error",
                    "message": message,
                }
            return event

        self.append_and_push_run_events([automation_event("started")])
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
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.append_and_push_run_events(
                [automation_event("failed", message=str(exc))]
            )
            raise

        output_item: JSONObject = {
            "type": "function_call_output",
            "call_id": call_id,
            "output": strict_json_dumps(output, compact=True),
        }
        self.append_and_push_run_events([automation_event("completed", output=output)])
        self.append_context_items([output_item])
        return output_item

    def push_run_events(self, events: Sequence[JSONObject]) -> None:
        """Queue structured run events for `StreamRunEvents` clients."""
        for event in events:
            self._events.emit(event)

    def append_and_push_run_events(self, events: list[JSONObject]) -> None:
        """Append run events to context and push them to `StreamRunEvents` clients."""
        if not events:
            return
        append_items(self._context, events)
        self.push_run_events(events)

    def append_context_items(self, items: list[JSONObject]) -> None:
        """Append OpenResponses items to the AgentApp context."""
        append_items(self._context, items)

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
        deadline = time.monotonic() + _DEFAULT_MODEL_REPLY_TIMEOUT
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
                raise TimeoutError("Timed out waiting for model response.")

            time.sleep(min(_DEFAULT_MODEL_REPLY_POLL_INTERVAL, remaining))


def _is_json_object_list(obj: JSONValue) -> bool:
    """Check if the given object is a list of JSON objects."""
    return isinstance(obj, list) and all(isinstance(item, dict) for item in obj)
