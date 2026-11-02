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
"""Flower error translation for protobuf-over-HTTP clients."""

import httpx

from flwr.supercore.error import FlowerError
from flwr.supercore.protobuf.client import ProtobufCall, ProtobufRequestContext


class FlowerErrorHttpInterceptor:
    """Raise structured HTTP errors as FlowerError instances.

    Add this interceptor explicitly to a protobuf client's interceptor list.
    Translated errors bypass the client's HTTP retry policy.
    """

    def intercept(
        self,
        context: ProtobufRequestContext,
        call_next: ProtobufCall,
    ) -> httpx.Response:
        """Restore Flower errors while preserving other HTTP exceptions."""
        try:
            response = call_next(context)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if error := FlowerError.from_json(exc.response.text):
                raise error from exc
            raise
        return response
