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
# ===============================================================================
"""Client interceptors for Flower CLI metadata."""

import httpx

from flwr.supercore.constant import FLWR_CLIENT_METADATA_KEY
from flwr.supercore.protobuf.client import ProtobufCall, ProtobufRequestContext


class CliClientHttpInterceptor:
    """Attach the CLI client identifier to protobuf-over-HTTP requests."""

    def intercept(
        self,
        context: ProtobufRequestContext,
        call_next: ProtobufCall,
    ) -> httpx.Response:
        """Add the CLI client identifier to an HTTP call."""
        context.request.headers[FLWR_CLIENT_METADATA_KEY] = "cli"
        return call_next(context)
