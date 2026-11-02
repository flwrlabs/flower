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
"""SuperExec authentication interceptor for protobuf-over-HTTP clients."""

from collections.abc import Collection

import httpx

from flwr.supercore.auth import create_superexec_auth_metadata, derive_auth_secret
from flwr.supercore.protobuf.client import ProtobufCall, ProtobufRequestContext

from .utils import add_headers


class SuperExecAuthHttpInterceptor:
    """Attach SuperExec HMAC authentication headers to HTTP requests."""

    def __init__(
        self,
        *,
        master_secret: bytes,
        protected_methods: Collection[str],
    ) -> None:
        self._auth_secret = derive_auth_secret(master_secret)
        self._protected_methods = frozenset(protected_methods)

    def intercept(
        self,
        context: ProtobufRequestContext,
        call_next: ProtobufCall,
    ) -> httpx.Response:
        """Sign protected requests before sending them."""
        if context.rpc_method in self._protected_methods:
            add_headers(
                context.request,
                create_superexec_auth_metadata(
                    auth_secret=self._auth_secret,
                    method=context.rpc_method,
                    request=context.message,
                ),
            )
        return call_next(context)
