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
"""Runtime-version interceptor for protobuf-over-HTTP clients."""

from logging import WARN

import requests

from flwr.common.logger import log
from flwr.supercore.constant import VERSION_INCOMPATIBILITY_MESSAGE_METADATA_KEY
from flwr.supercore.exit import ExitCode, flwr_exit
from flwr.supercore.protobuf.client import ProtobufCall, ProtobufRequestContext
from flwr.supercore.runtime_version_compatibility import (
    RuntimeVersionMetadata,
    get_runtime_version_incompatibility_exit_message,
)

from .utils import add_headers


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
        add_headers(context.request, dict(self._metadata.as_metadata()))
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
