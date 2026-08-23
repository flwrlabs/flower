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
"""SuperLink FastAPI extension hooks."""

import asyncio
import threading
from asyncio import AbstractEventLoop
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from copy import deepcopy
from importlib import import_module
from logging import WARNING
from types import ModuleType
from typing import Any, Literal, cast

from fastapi import FastAPI
from starlette.middleware import Middleware

from flwr.common.logger import log
from flwr.supercore.run import Run

SuperLinkLifespanContext = Callable[
    [FastAPI], AbstractAsyncContextManager[Mapping[str, Any] | None]
]
# API transport does not identify the caller (CLI and web UI can use either
# gRPC or HTTP). API-created runs therefore use ``unknown`` unless a trusted
# Flower-owned integration supplies a more specific source.
RunStartSource = Literal["cli", "web_ui", "automation", "unknown"]
_SGXT_MODULE = "flwr.ee.superlink.extensions"
_NOTIFICATION_LOOP: AbstractEventLoop | None = None
_NOTIFICATION_TASK_CAPACITY = 1000
_NOTIFICATION_CALLBACK_CAPACITY = 4
_NOTIFICATION_CALLBACK_TIMEOUT_SECONDS = 1.0
_NOTIFICATION_TASKS: set[asyncio.Task[None]] = set()
_NOTIFICATION_QUEUE_SLOTS = threading.BoundedSemaphore(_NOTIFICATION_TASK_CAPACITY)
_NOTIFICATION_CALLBACK_SLOTS = asyncio.Semaphore(_NOTIFICATION_CALLBACK_CAPACITY)


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


def set_notification_loop(loop: AbstractEventLoop) -> None:
    """Set the event loop used to dispatch extension notifications.

    SuperLink owns one FastAPI event loop while its synchronous gRPC handlers
    run in worker threads. Scheduling onto that loop provides the lifecycle
    boundary; synchronous extension callbacks are then isolated in a bounded
    set of daemon callback threads so they cannot block the event loop.
    """
    global _NOTIFICATION_LOOP  # pylint: disable=global-statement
    _NOTIFICATION_LOOP = loop


def clear_notification_loop() -> None:
    """Clear the event loop used for extension notification dispatch."""
    global _NOTIFICATION_LOOP  # pylint: disable=global-statement
    _NOTIFICATION_LOOP = None
    for task in _NOTIFICATION_TASKS:
        task.cancel()
    _NOTIFICATION_TASKS.clear()


def _invoke_extension(
    callback_name: str,
    callback_args: tuple[Any, ...],
    label: str,
) -> None:
    """Discover and invoke one optional extension callback."""
    try:
        sgxt = _try_import_sgxt()
        if sgxt is None:
            return
        callback = cast(Callable[..., None] | None, getattr(sgxt, callback_name, None))
        if callback is not None:
            # Take the snapshot at the final dispatch boundary so importing an
            # extension and copying a mutable Run never happens on an API path.
            callback(*deepcopy(callback_args))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        log(
            WARNING,
            "%s extension notification failed: %s.",
            label,
            type(exc).__name__,
            exc_info=exc,
        )


async def _dispatch_extension(
    callback_name: str,
    callback_args: tuple[Any, ...],
    label: str,
) -> None:
    """Run a synchronous callback without blocking the service event loop."""
    # Pending tasks wait here, up to _NOTIFICATION_TASK_CAPACITY, instead of
    # dropping notifications merely because all callback slots are busy.
    await _NOTIFICATION_CALLBACK_SLOTS.acquire()

    loop = asyncio.get_running_loop()
    completed = asyncio.Event()

    def invoke_callback() -> None:
        try:
            _invoke_extension(callback_name, callback_args, label)
        finally:

            def callback_done() -> None:
                _NOTIFICATION_CALLBACK_SLOTS.release()
                completed.set()

            loop.call_soon_threadsafe(callback_done)

    callback_thread = threading.Thread(  # pylint: disable=consider-using-with
        target=invoke_callback,
        name="superlink-extension-callback",
        daemon=True,
    )
    try:
        callback_thread.start()
    except Exception:  # pylint: disable=broad-exception-caught
        _NOTIFICATION_CALLBACK_SLOTS.release()
        log(WARNING, "%s extension notification could not start.", label)
        return

    try:
        await asyncio.wait_for(
            completed.wait(), timeout=_NOTIFICATION_CALLBACK_TIMEOUT_SECONDS
        )
    except TimeoutError:
        log(
            WARNING,
            "%s extension notification timed out after %.1f seconds.",
            label,
            _NOTIFICATION_CALLBACK_TIMEOUT_SECONDS,
        )


def _schedule_extension(
    callback_name: str,
    callback_args: tuple[Any, ...],
    label: str,
) -> None:
    """Schedule one bounded extension task on the service event loop."""
    if _NOTIFICATION_LOOP is not asyncio.get_running_loop():
        _NOTIFICATION_QUEUE_SLOTS.release()
        return
    task = asyncio.create_task(_dispatch_extension(callback_name, callback_args, label))
    _NOTIFICATION_TASKS.add(task)

    def task_done(completed_task: asyncio.Task[None]) -> None:
        _NOTIFICATION_TASKS.discard(completed_task)
        _NOTIFICATION_QUEUE_SLOTS.release()

    task.add_done_callback(task_done)


def _notify_extension(
    callback_name: str,
    callback_args: tuple[Any, ...],
    label: str,
) -> None:
    """Dispatch one optional extension callback outside API request handlers."""
    loop = _NOTIFICATION_LOOP
    if loop is not None and loop.is_running():
        if not _NOTIFICATION_QUEUE_SLOTS.acquire(  # pylint: disable=consider-using-with
            blocking=False
        ):
            log(WARNING, "%s extension notification queue is full.", label)
            return
        try:
            loop.call_soon_threadsafe(
                _schedule_extension, callback_name, callback_args, label
            )
            return
        except RuntimeError:
            # The service can begin shutdown between checking ``is_running`` and
            # scheduling. Fall back to the direct path so notifications are not
            # silently lost during orderly shutdown.
            _NOTIFICATION_QUEUE_SLOTS.release()

    # Direct calls are used by standalone handlers and unit tests which do not
    # own the combined SuperLink lifespan. The normal service path always has a
    # loop configured before accepting requests.
    _invoke_extension(callback_name, callback_args, label)


def notify_run_started(run: Run, source: RunStartSource) -> None:
    """Notify an optional extension after a run has been created successfully.

    In the combined SuperLink service the callback is scheduled on the existing
    FastAPI event loop, so it does not delay the gRPC or HTTP response. It
    receives a snapshot so it cannot mutate the run stored by SuperLink.
    Optional extension failures are logged and ignored. Standalone callers
    without a configured loop must provide their own non-blocking boundary.
    """
    _notify_extension("on_run_started", (run, source), "Run-started")
