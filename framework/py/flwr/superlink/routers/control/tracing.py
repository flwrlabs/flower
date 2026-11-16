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
"""Tracing dependency for authenticated run creation."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request

from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.tracing import TraceSpan, trace_span
from flwr.superlink.dependencies.account import get_account


async def trace_run_creation(
    request: Request, _account: Annotated[AccountInfo, Depends(get_account)]
) -> AsyncIterator[TraceSpan]:
    """Continue the incoming context only after account authorization."""
    with trace_span(
        "run.create",
        traceparent=request.headers.get("traceparent", ""),
        attributes={"flwr.component": "superlink"},
    ) as span:
        yield span


RunCreationTracingDependency = Annotated[
    TraceSpan, Depends(trace_run_creation, scope="function")
]
