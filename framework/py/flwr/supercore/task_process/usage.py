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
"""Task usage recording helpers for trusted task runtimes."""

from __future__ import annotations

from typing import Protocol

from flwr.proto.appio_pb2 import RecordTaskUsageRequest  # pylint: disable=E0611
from flwr.proto.task_pb2 import TaskUsage  # pylint: disable=E0611
from flwr.supercore.typing import JSONObject

MODEL_INFERENCE_USAGE_TYPE = "model_inference"
WEB_SEARCH_USAGE_TYPE = "web_search"


class _TaskUsageStub(Protocol):
    """AppIo stub surface needed to record task usage."""

    def RecordTaskUsage(  # pylint: disable=invalid-name
        self, request: RecordTaskUsageRequest
    ) -> object:
        """Record task usage through AppIo."""


class TaskUsageRecorder:
    """Record task usage through the authenticated AppIo task token."""

    def __init__(self, stub: _TaskUsageStub) -> None:
        self._stub = stub

    def record(self, usage: TaskUsage) -> None:
        """Record one task usage payload."""
        self._stub.RecordTaskUsage(
            RecordTaskUsageRequest(
                task_usage=TaskUsage(
                    usage_type=usage.usage_type,
                    provider=usage.provider,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    total_tokens=usage.total_tokens,
                )
            )
        )


def task_usage_from_open_response(
    response: JSONObject, *, provider: str
) -> TaskUsage | None:
    """Extract Open Responses-compatible token usage from a response."""
    raw_usage = response.get("usage")
    if not isinstance(raw_usage, dict):
        return None

    usage = TaskUsage(usage_type=MODEL_INFERENCE_USAGE_TYPE, provider=provider)
    has_tokens = False
    for response_field, proto_field in (
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        value = raw_usage.get(response_field)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            setattr(usage, proto_field, value)
            has_tokens = True

    return usage if has_tokens else None
