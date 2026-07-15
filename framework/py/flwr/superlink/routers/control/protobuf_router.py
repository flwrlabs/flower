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
"""Control API protobuf router."""

from collections.abc import Callable

from fastapi import Request
from google.protobuf.message import Message
from starlette.datastructures import State

from flwr.supercore.protobuf.routing import ProtobufRouter

from .event_log import ControlEventLogger
from .license import ControlLicenseChecker


class ControlProtobufRouter(ProtobufRouter):
    """Protobuf router that composes the Control request components."""

    async def _call_handler(
        self,
        func: Callable[..., object],
        http_request: Request[State],
        proto_request: Message,
        dependency_values: dict[str, object],
    ) -> object:
        """Run the license and event-log components around a handler call."""
        ControlLicenseChecker.check(http_request)
        return await ControlEventLogger.call(
            func, http_request, proto_request, dependency_values
        )
