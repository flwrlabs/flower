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
"""FastAPI dependency for Control API event logging."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from google.protobuf.message import Message
from starlette.concurrency import run_in_threadpool

from flwr.common.event_log_plugin import EventLogWriterPlugin
from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.protobuf.translation import get_protobuf_request


async def log_control_event(
    request: Request,
    protobuf_request: Annotated[Message, Depends(get_protobuf_request)],
) -> AsyncIterator[None]:
    """Write events before and after a Control API handler call."""
    event_log_plugin: EventLogWriterPlugin | None = getattr(
        request.app.state, "control_event_log_plugin", None
    )
    if event_log_plugin is None:
        yield
        return

    # Authentication runs before dependency resolution and stores the account,
    # except for unauthenticated Control routes where the actor remains unknown.
    account_info = getattr(request.state, "account", None)
    if not isinstance(account_info, AccountInfo):
        account_info = None

    def write_before_event() -> None:
        """Compose and write the event preceding handler execution."""
        event_log_plugin.write_log(
            event_log_plugin.compose_log_before_event(
                request=protobuf_request,
                context=request,
                account_info=account_info,
                method_name=request.url.path,
            )
        )

    def write_after_event(result: Message | BaseException | None) -> None:
        """Compose and write the event following handler execution."""
        event_log_plugin.write_log(
            event_log_plugin.compose_log_after_event(
                request=protobuf_request,
                context=request,
                account_info=account_info,
                method_name=request.url.path,
                response=result,
            )
        )

    await run_in_threadpool(write_before_event)
    try:
        yield
    except BaseException as exc:
        # Record handler and dependency failures before propagating them to the
        # outer HTTP error-translation middleware.
        await run_in_threadpool(write_after_event, exc)
        raise

    result = getattr(request.state, "protobuf_response", None)
    if isinstance(result, Message):
        await run_in_threadpool(write_after_event, result)
    else:
        # Streaming response event logging is not yet implemented.
        pass
