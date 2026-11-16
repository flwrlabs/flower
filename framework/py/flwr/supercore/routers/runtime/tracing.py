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
"""Best-effort tracing integration for the Runtime HTTP API."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request

from flwr.supercore.dependencies.runtime import TaskDependency
from flwr.supercore.tracing import trace_span, validate_traceparent


async def trace_task_completion(
    request: Request, task: TaskDependency
) -> AsyncIterator[None]:
    """Scope completion tracing to the authenticated handler."""
    try:
        carrier = validate_traceparent(request.headers.get("traceparent"))
    except Exception:  # pylint: disable=broad-exception-caught
        carrier = ""
    with trace_span(
        "task.complete",
        traceparent=carrier,
        attributes={
            "flwr.task_id": str(task.task_id),
            "flwr.run_id": str(task.run_id),
        },
    ):
        yield


TaskCompletionTracingDependency = Annotated[
    None, Depends(trace_task_completion, scope="function")
]
