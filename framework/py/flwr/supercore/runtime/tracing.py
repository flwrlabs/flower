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
"""Optional tracing interceptor owned by Runtime API callers."""

import httpx

from flwr.supercore.protobuf.client import ProtobufCall, ProtobufRequestContext
from flwr.supercore.tracing import current_traceparent


class RuntimeTracingInterceptor:
    """Propagate active trace context in outbound Runtime requests."""

    def intercept(
        self, context: ProtobufRequestContext, call_next: ProtobufCall
    ) -> httpx.Response:
        """Keep telemetry failures separate from transport and application errors."""
        try:
            if carrier := current_traceparent():
                context.request.headers["traceparent"] = carrier
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        return call_next(context)
