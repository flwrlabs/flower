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
"""Handle model tasks."""


from __future__ import annotations

import time
from typing import cast

from flwr.common.serde import message_from_proto, message_to_proto
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    PullTaskMessageRequest,
    PushTaskEventsRequest,
    PushTaskMessageRequest,
)
from flwr.proto.task_pb2 import TaskEvent  # pylint: disable=E0611
from flwr.supercore.json_message.model_message import ModelRequest, ModelResponse
from flwr.supercore.runtime import RuntimeHttpClient
from flwr.supercore.runtime_timing import (
    RuntimeTimingErrorKind,
    RuntimeTimingOutcome,
    emit_runtime_timing,
)
from flwr.supercore.task_process.usage import TaskUsageRecorder
from flwr.supercore.typing import JSONObject
from flwr.supercore.utils import strict_json_dumps

from .provider import ModelProviderError, invoke_model_provider

_DEFAULT_TASK_EVENT_BATCH_SIZE = 16
_TEXT_DELTA_EVENTS = frozenset(
    {"response.output_text.delta", "response.reasoning_summary_text.delta"}
)


def handle_task(  # pylint: disable=too-many-locals,too-many-statements
    client: RuntimeHttpClient, task_id: int, run_id: int
) -> None:
    """Run one model task request."""
    request_message = _pull_model_request(client)
    is_stream = request_message.payload.get("stream") is True
    if request_message.metadata.src_task_id is None:
        raise RuntimeError("Model request source task is not set.")
    parent_task_id = request_message.metadata.src_task_id

    emit_runtime_timing(
        "runtime.application.input.received",
        component="model_task",
        run_id=run_id,
        task_id=task_id,
        parent_task_id=parent_task_id,
        root_task_id=parent_task_id,
        task_type="flwr-model",
        process_mode="new",
    )
    emit_runtime_timing(
        "runtime.model.execution.started",
        component="model_task",
        run_id=run_id,
        task_id=task_id,
        parent_task_id=parent_task_id,
        root_task_id=parent_task_id,
        task_type="flwr-model",
        process_mode="new",
    )

    def _push_model_response(response: JSONObject) -> None:
        """Push a ModelResponse back to the requesting task."""
        message = ModelResponse(
            dst_task_id=cast(int, request_message.metadata.src_task_id),
            response=response,
            reply_to_message_id=request_message.metadata.message_id,
        )
        message.metadata.__dict__["_run_id"] = run_id
        message.metadata.src_task_id = task_id
        message.metadata.__dict__["_message_id"] = message.object_id
        client.PushTaskMessage(
            PushTaskMessageRequest(message=message_to_proto(message))
        )

    # Stream events are exposed through Control.StreamRunEvents.
    events: list[TaskEvent] = []
    first_provider_event_received = False
    first_event_flush_finished = False
    publisher_failed = False
    first_text_event_flushed = False

    def _flush_events() -> None:
        """Push buffered stream events."""
        nonlocal first_event_flush_finished, publisher_failed
        if not is_stream or not events:
            return
        is_first_flush = not first_event_flush_finished
        if is_first_flush:
            emit_runtime_timing(
                "runtime.model.first_event.flush.started",
                component="model_task",
                run_id=run_id,
                task_id=task_id,
                parent_task_id=parent_task_id,
                root_task_id=parent_task_id,
                task_type="flwr-model",
                process_mode="new",
            )
        try:
            client.PushTaskEvents(PushTaskEventsRequest(events=events))
        except Exception:  # pylint: disable=broad-exception-caught
            publisher_failed = True
            emit_runtime_timing(
                "runtime.events.publish.failed",
                component="model_task",
                run_id=run_id,
                task_id=task_id,
                parent_task_id=parent_task_id,
                root_task_id=parent_task_id,
                task_type="flwr-model",
                outcome="error",
                error_kind="publisher",
                process_mode="new",
            )
            raise
        if is_first_flush:
            first_event_flush_finished = True
            emit_runtime_timing(
                "runtime.model.first_event.flush.finished",
                component="model_task",
                run_id=run_id,
                task_id=task_id,
                parent_task_id=parent_task_id,
                root_task_id=parent_task_id,
                task_type="flwr-model",
                outcome="ok",
                process_mode="new",
            )
        events.clear()

    def _buffer_event(event: JSONObject) -> None:
        """Buffer one Open Responses stream event."""
        nonlocal first_provider_event_received, first_text_event_flushed
        if not is_stream:
            return
        if not first_provider_event_received:
            first_provider_event_received = True
            emit_runtime_timing(
                "runtime.model.provider.first_event.received",
                component="model_task",
                run_id=run_id,
                task_id=task_id,
                parent_task_id=parent_task_id,
                root_task_id=parent_task_id,
                task_type="flwr-model",
                process_mode="new",
            )
        encoded = strict_json_dumps(event, compact=True)
        events.append(TaskEvent(event=cast(str, event["type"]), data=encoded))
        if event["type"] in _TEXT_DELTA_EVENTS and not first_text_event_flushed:
            _flush_events()
            first_text_event_flushed = True
        elif len(events) >= _DEFAULT_TASK_EVENT_BATCH_SIZE:
            _flush_events()

    response: JSONObject | None = None
    provider_outcome: RuntimeTimingOutcome = "error"
    provider_error_kind: RuntimeTimingErrorKind | None = "unknown"
    model_outcome: RuntimeTimingOutcome = "error"
    model_error_kind: RuntimeTimingErrorKind | None = "unknown"
    try:
        emit_runtime_timing(
            "runtime.model.provider.request.started",
            component="model_task",
            run_id=run_id,
            task_id=task_id,
            parent_task_id=parent_task_id,
            root_task_id=parent_task_id,
            task_type="flwr-model",
            process_mode="new",
        )
        try:
            response = invoke_model_provider(
                request_message.payload,
                on_stream_event=_buffer_event,
                usage_recorder=TaskUsageRecorder(client),
            )
        except Exception as ex:  # pylint: disable=broad-exception-caught
            provider_error_kind = "publisher" if publisher_failed else "provider"
            response = _make_error_response(ex)
        else:
            provider_outcome = "ok"
            provider_error_kind = None
        finally:
            emit_runtime_timing(
                "runtime.model.provider.stream.finished",
                component="model_task",
                run_id=run_id,
                task_id=task_id,
                parent_task_id=parent_task_id,
                root_task_id=parent_task_id,
                task_type="flwr-model",
                outcome=provider_outcome,
                error_kind=provider_error_kind,
                process_mode="new",
            )
            try:
                # Flush partial batches after the provider stream ends or fails.
                _flush_events()
            except Exception:  # pylint: disable=broad-exception-caught
                provider_outcome = "error"
                provider_error_kind = "publisher"
                raise
            # Preserve the existing error-reply behavior for provider failures.
            if response is not None:
                try:
                    _push_model_response(response)
                except Exception:  # pylint: disable=broad-exception-caught
                    provider_error_kind = "publisher"
                    raise
        model_outcome = "ok"
        model_error_kind = None
    except Exception:  # pylint: disable=broad-exception-caught
        model_error_kind = provider_error_kind or "unknown"
        raise
    finally:
        emit_runtime_timing(
            "runtime.model.execution.finished",
            component="model_task",
            run_id=run_id,
            task_id=task_id,
            parent_task_id=parent_task_id,
            root_task_id=parent_task_id,
            task_type="flwr-model",
            outcome=model_outcome,
            error_kind=model_error_kind if model_outcome == "error" else None,
            process_mode="new",
        )


def _pull_model_request(client: RuntimeHttpClient) -> ModelRequest:
    """Pull one model request, waiting until it becomes available."""
    # Keep polling until flwr-agentapp produces a request. If it exits, cleanup
    # forces flwr-model to stop, with auth handling revoked tokens.
    while True:
        pull_response = client.PullTaskMessage(PullTaskMessageRequest(limit=1))
        messages = [message_from_proto(message) for message in pull_response.messages]
        if messages:
            return ModelRequest.from_message(messages[0])
        time.sleep(1)  # Wait for 1 second before trying again.


def _make_error_response(ex: Exception) -> JSONObject:
    """Create a JSON error response from an exception."""
    error_code = "internal_error"
    if isinstance(ex, ModelProviderError):
        error_code = "model_provider_error"
    return {
        "object": "response",
        "status": "failed",
        "error": {
            "code": error_code,
            "message": str(ex),
        },
        "output": [],
    }
