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
"""Task-token interceptor for protobuf-over-HTTP clients."""

import httpx

from flwr.supercore.constant import TASK_TOKEN_HEADER
from flwr.supercore.protobuf.client import ProtobufCall, ProtobufRequestContext

from .utils import add_headers


class RuntimeTokenHttpInterceptor:
    """Attach a Runtime task token to HTTP requests."""

    def __init__(self, token: str) -> None:
        if not token:
            raise ValueError("`token` must be a non-empty string")
        self._token = token

    def intercept(
        self,
        context: ProtobufRequestContext,
        call_next: ProtobufCall,
    ) -> httpx.Response:
        """Add the task-token header before sending the request."""
        add_headers(context.request, {TASK_TOKEN_HEADER: self._token})
        return call_next(context)
