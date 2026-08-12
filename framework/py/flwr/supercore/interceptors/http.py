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
"""Reusable protobuf-over-HTTP client interceptors."""

from __future__ import annotations

from collections.abc import Collection
from logging import WARN

import requests

from flwr.common.logger import log
from flwr.supercore.auth import create_superexec_auth_metadata, derive_auth_secret
from flwr.supercore.constant import VERSION_INCOMPATIBILITY_MESSAGE_METADATA_KEY
from flwr.supercore.exit import ExitCode, flwr_exit
from flwr.supercore.protobuf.client import ProtobufCall, ProtobufRequestContext
from flwr.supercore.runtime_version_compatibility import (
    RuntimeVersionMetadata,
    get_runtime_version_incompatibility_exit_message,
)


def _add_headers(
    request: requests.PreparedRequest,
    headers: dict[str, str],
) -> None:
    """Add headers while rejecting values already provided by another layer."""
    duplicates = {name for name in headers if name in request.headers}
    if duplicates:
        raise RuntimeError(
            f"HTTP request already contains headers: {', '.join(sorted(duplicates))}"
        )
    request.headers.update(headers)


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
    ) -> requests.Response:
        """Sign protected requests before sending them."""
        if context.method in self._protected_methods:
            _add_headers(
                context.request,
                create_superexec_auth_metadata(
                    auth_secret=self._auth_secret,
                    method=context.method,
                    request=context.message,
                ),
            )
        return call_next(context)


class RuntimeVersionHttpInterceptor:
    """Exchange Flower runtime-version information over HTTP."""

    def __init__(self, component_name: str) -> None:
        self._metadata = RuntimeVersionMetadata.from_local_component(component_name)

    def intercept(
        self,
        context: ProtobufRequestContext,
        call_next: ProtobufCall,
    ) -> requests.Response:
        """Add local version headers and handle compatibility responses."""
        _add_headers(context.request, dict(self._metadata.as_metadata()))
        response = call_next(context)

        if incompatibility_message := response.headers.get(
            VERSION_INCOMPATIBILITY_MESSAGE_METADATA_KEY
        ):
            log(WARN, incompatibility_message)

        if not response.ok and (
            exit_message := get_runtime_version_incompatibility_exit_message(
                response.text
            )
        ):
            flwr_exit(ExitCode.RUNTIME_VERSION_INCOMPATIBLE, exit_message)

        return response
