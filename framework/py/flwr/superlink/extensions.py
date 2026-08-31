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
"""SuperLink extension hooks."""

from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from copy import deepcopy
from importlib import import_module
from logging import WARNING
from types import ModuleType
from typing import Any, Final, Literal, cast

from fastapi import FastAPI
from starlette.middleware import Middleware

from flwr.common.logger import log
from flwr.supercore.run import Run
from flwr.superlink.run_source import RunSource

SuperLinkLifespanContext = Callable[
    [FastAPI], AbstractAsyncContextManager[Mapping[str, Any] | None]
]
RESULT_DELIVERY_CHANNEL_LOGS: Final = "logs"
RESULT_DELIVERY_CHANNEL_CHAT: Final = "chat"
ResultDeliveryChannel = Literal["logs", "chat"]
_SGXT_MODULE = "flwr.ee.superlink.extensions"


def _try_import_sgxt() -> ModuleType | None:
    """Return the SuperGrid Extensions module when it is installed."""
    try:
        return import_module(_SGXT_MODULE)
    except ModuleNotFoundError as exc:
        # Ignore only an absent SuperGrid Extensions package or module. Missing
        # dependencies imported by an existing extension must still fail loudly.
        if exc.name is None or not (
            exc.name == _SGXT_MODULE or _SGXT_MODULE.startswith(f"{exc.name}.")
        ):
            raise
        return None


def configure_app(app: FastAPI) -> None:
    """Configure SuperLink FastAPI extensions."""
    sgxt = _try_import_sgxt()
    if sgxt is None:
        return

    configure_sgxt_app = cast(
        Callable[[FastAPI], None] | None,
        getattr(sgxt, "configure_app", None),
    )
    if configure_sgxt_app is not None:
        configure_sgxt_app(app)


def get_middleware() -> tuple[Middleware, ...]:
    """Return extension middleware in request execution order."""
    sgxt = _try_import_sgxt()
    if sgxt is None:
        return ()

    get_sgxt_middleware = cast(
        Callable[[], tuple[Middleware, ...]] | None,
        getattr(sgxt, "get_middleware", None),
    )
    if get_sgxt_middleware is None:
        # Compatibility with SuperGrid Extensions versions predating this hook.
        return ()
    return get_sgxt_middleware()


def get_lifespan_contexts() -> tuple[SuperLinkLifespanContext, ...]:
    """Return SuperLink FastAPI lifespan contexts."""
    sgxt = _try_import_sgxt()
    if sgxt is None:
        return ()

    get_sgxt_lifespan_contexts = cast(
        Callable[[], tuple[SuperLinkLifespanContext, ...]] | None,
        getattr(sgxt, "get_lifespan_contexts", None),
    )
    if get_sgxt_lifespan_contexts is None:
        return ()
    return get_sgxt_lifespan_contexts()


def notify_run_started(run: Run, source: RunSource) -> None:
    """Notify an optional extension after a run has been persisted.

    The callback is synchronous by design. Extensions must keep this hook
    non-blocking and best effort; the Flower framework does not create a
    background thread or event loop for it. The run snapshot is copied before
    handing it to the extension so the callback cannot mutate the object used
    to build the successful StartRun response. The source is also best-effort
    caller attribution and must not be used for authorization decisions.
    """
    try:
        sgxt = _try_import_sgxt()
        if sgxt is None:
            return

        on_run_started = cast(
            Callable[[Run, RunSource], None] | None,
            getattr(sgxt, "on_run_started", None),
        )
        if on_run_started is not None:
            on_run_started(deepcopy(run), source)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        log(
            WARNING,
            "Run-start extension notification failed: %s.",
            type(exc).__name__,
            exc_info=exc,
        )


def notify_result_delivered(
    run: Run,
    flwr_aid: str,
    channel: ResultDeliveryChannel,
) -> None:
    """Notify an optional extension after a result request was accepted.

    The callback is synchronous by design. Extensions must keep this hook
    non-blocking and best effort; the Flower framework does not create a
    background thread for it. The run snapshot is copied before handing it to
    the extension so the callback cannot mutate SuperLink state.
    """
    try:
        sgxt = _try_import_sgxt()
        if sgxt is None:
            return

        on_result_delivered = cast(
            Callable[[Run, str, ResultDeliveryChannel], None] | None,
            getattr(sgxt, "on_result_delivered", None),
        )
        if on_result_delivered is not None:
            on_result_delivered(deepcopy(run), flwr_aid, channel)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        log(
            WARNING,
            "Result-delivered extension notification failed: %s.",
            type(exc).__name__,
            exc_info=exc,
        )
