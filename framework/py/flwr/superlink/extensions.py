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
from logging import WARNING
from typing import Any, Literal

from fastapi import FastAPI
from starlette.middleware import Middleware

from flwr.common.logger import log
from flwr.supercore.run import Run
from flwr.superlink.run_source import RunStartSource

SuperLinkLifespanContext = Callable[
    [FastAPI], AbstractAsyncContextManager[Mapping[str, Any] | None]
]
ResultDeliveryChannel = Literal["logs", "chat"]
_SGXT_MODULE = "flwr.ee.superlink.extensions"


def _is_sgxt_module_not_found(exc: ModuleNotFoundError) -> bool:
    """Return whether a missing module means that SGXT is unavailable."""
    return exc.name == _SGXT_MODULE or (
        exc.name is not None and _SGXT_MODULE.startswith(f"{exc.name}.")
    )


def configure_app(app: FastAPI) -> None:
    """Configure SuperLink FastAPI extensions."""
    try:
        # pylint: disable-next=import-outside-toplevel
        from flwr.ee.superlink.extensions import configure_app as _configure_sgxt_app
    except ModuleNotFoundError as exc:
        if _is_sgxt_module_not_found(exc):
            return
        raise

    configure_sgxt_app: Callable[[FastAPI], None]
    configure_sgxt_app = _configure_sgxt_app
    configure_sgxt_app(app)


def get_middleware() -> tuple[Middleware, ...]:
    """Return extension middleware in request execution order."""
    try:
        # pylint: disable-next=import-outside-toplevel
        from flwr.ee.superlink.extensions import get_middleware as _get_sgxt_middleware
    except ModuleNotFoundError as exc:
        if _is_sgxt_module_not_found(exc):
            return ()
        raise

    get_sgxt_middleware: Callable[[], tuple[Middleware, ...]]
    get_sgxt_middleware = _get_sgxt_middleware
    return get_sgxt_middleware()


def get_lifespan_contexts() -> tuple[SuperLinkLifespanContext, ...]:
    """Return SuperLink FastAPI lifespan contexts."""
    try:
        # pylint: disable-next=import-outside-toplevel
        from flwr.ee.superlink.extensions import (
            get_lifespan_contexts as _get_sgxt_lifespan_contexts,
        )
    except ModuleNotFoundError as exc:
        if _is_sgxt_module_not_found(exc):
            return ()
        raise

    get_sgxt_lifespan_contexts: Callable[[], tuple[SuperLinkLifespanContext, ...]]
    get_sgxt_lifespan_contexts = _get_sgxt_lifespan_contexts
    return get_sgxt_lifespan_contexts()


def notify_run_started(run: Run, source: RunStartSource) -> None:
    """Notify an optional extension after a run has been persisted.

    The callback is synchronous by design. Extensions must keep this hook
    non-blocking and best effort; the Flower framework does not create a
    background thread or event loop for it. The run snapshot is copied before
    handing it to the extension so the callback cannot mutate the object used
    to build the successful StartRun response. The source is also best-effort
    caller attribution and must not be used for authorization decisions.
    """
    try:
        try:
            # pylint: disable-next=import-outside-toplevel
            from flwr.ee.superlink.extensions import on_run_started as _on_run_started
        except ModuleNotFoundError as exc:
            if _is_sgxt_module_not_found(exc):
                return
            raise

        on_run_started: Callable[[Run, RunStartSource], None]
        on_run_started = _on_run_started
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
        try:
            # pylint: disable-next=import-outside-toplevel
            from flwr.ee.superlink.extensions import (
                on_result_delivered as _on_result_delivered,
            )
        except ModuleNotFoundError as exc:
            if _is_sgxt_module_not_found(exc):
                return
            raise

        on_result_delivered: Callable[[Run, str, ResultDeliveryChannel], None]
        on_result_delivered = _on_result_delivered
        on_result_delivered(deepcopy(run), flwr_aid, channel)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        log(
            WARNING,
            "Result-delivered extension notification failed: %s.",
            type(exc).__name__,
            exc_info=exc,
        )
