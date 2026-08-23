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
_NOTIFICATION_CALLBACK_SLOTS: asyncio.Semaphore | None = None
_NOTIFICATION_CALLBACK_EVENTS: set[threading.Event] = set()
_NOTIFICATION_CALLBACK_EVENTS_LOCK = threading.Lock()
_NOTIFICATION_PENDING_SUBMISSIONS = 0
_NOTIFICATION_PENDING_SUBMISSIONS_LOCK = threading.Lock()
_NOTIFICATION_STANDALONE_CALLBACK_SLOTS = threading.BoundedSemaphore(
    _NOTIFICATION_CALLBACK_CAPACITY
)


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
    global _NOTIFICATION_LOOP, _NOTIFICATION_CALLBACK_SLOTS  # pylint: disable=global-statement
    _NOTIFICATION_LOOP = loop
    _NOTIFICATION_CALLBACK_SLOTS = asyncio.Semaphore(_NOTIFICATION_CALLBACK_CAPACITY)


def clear_notification_loop() -> None:
    """Clear the event loop used for extension notification dispatch."""
    global _NOTIFICATION_LOOP, _NOTIFICATION_CALLBACK_SLOTS  # pylint: disable=global-statement
    _NOTIFICATION_LOOP = None
    _NOTIFICATION_CALLBACK_SLOTS = None
    for task in _NOTIFICATION_TASKS:
        task.cancel()
    _NOTIFICATION_TASKS.clear()


async def shutdown_notification_loop() -> None:
    """Stop dispatching and wait briefly for callback workers to finish."""
    await _drain_notification_submissions()
    clear_notification_loop()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _NOTIFICATION_CALLBACK_TIMEOUT_SECONDS
    while True:
        with _NOTIFICATION_CALLBACK_EVENTS_LOCK:
            pending_callbacks = tuple(_NOTIFICATION_CALLBACK_EVENTS)
        if not pending_callbacks:
            return

        remaining = deadline - loop.time()
        if remaining <= 0:
            log(
                WARNING,
                "Some extension callbacks did not finish during shutdown.",
            )
            return
        await asyncio.sleep(min(0.01, remaining))


async def _drain_notification_submissions() -> None:
    """Let thread-safe loop submissions become tracked tasks before teardown."""
    had_pending_submission = False
    while True:
        with _NOTIFICATION_PENDING_SUBMISSIONS_LOCK:
            if _NOTIFICATION_PENDING_SUBMISSIONS == 0:
                break
            had_pending_submission = True
        await asyncio.sleep(0)
    if had_pending_submission:
        # Give tasks created by the ready-queue submissions one turn to start
        # before clear_notification_loop cancels queued tasks.
        await asyncio.sleep(0)


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
    callback_slots = _NOTIFICATION_CALLBACK_SLOTS
    if callback_slots is None:
        return
    await callback_slots.acquire()

    loop = asyncio.get_running_loop()
    completed = threading.Event()
    with _NOTIFICATION_CALLBACK_EVENTS_LOCK:
        _NOTIFICATION_CALLBACK_EVENTS.add(completed)

    def invoke_callback() -> None:
        try:
            _invoke_extension(callback_name, callback_args, label)
        finally:
            completed.set()
            with _NOTIFICATION_CALLBACK_EVENTS_LOCK:
                _NOTIFICATION_CALLBACK_EVENTS.discard(completed)
            # Release the loop-bound semaphore while the service loop is
            # alive. If shutdown already closed it, the semaphore belongs to
            # a retired lifespan and no longer needs to be released.
            try:
                loop.call_soon_threadsafe(callback_slots.release)
            except RuntimeError:
                pass

    callback_thread = threading.Thread(  # pylint: disable=consider-using-with
        target=invoke_callback,
        name="superlink-extension-callback",
        daemon=True,
    )
    try:
        callback_thread.start()
    except Exception:  # pylint: disable=broad-exception-caught
        with _NOTIFICATION_CALLBACK_EVENTS_LOCK:
            _NOTIFICATION_CALLBACK_EVENTS.discard(completed)
        callback_slots.release()
        log(WARNING, "%s extension notification could not start.", label)
        return

    deadline = loop.time() + _NOTIFICATION_CALLBACK_TIMEOUT_SECONDS
    while not completed.is_set():
        remaining = deadline - loop.time()
        if remaining <= 0:
            log(
                WARNING,
                "%s extension notification timed out after %.1f seconds.",
                label,
                _NOTIFICATION_CALLBACK_TIMEOUT_SECONDS,
            )
            return
        await asyncio.sleep(min(0.01, remaining))


def _schedule_extension(
    callback_name: str,
    callback_args: tuple[Any, ...],
    label: str,
) -> None:
    """Schedule one bounded extension task on the service event loop."""
    global _NOTIFICATION_PENDING_SUBMISSIONS  # pylint: disable=global-statement
    with _NOTIFICATION_PENDING_SUBMISSIONS_LOCK:
        _NOTIFICATION_PENDING_SUBMISSIONS -= 1
    if _NOTIFICATION_LOOP is not asyncio.get_running_loop():
        _NOTIFICATION_QUEUE_SLOTS.release()
        return
    task = asyncio.create_task(_dispatch_extension(callback_name, callback_args, label))
    _NOTIFICATION_TASKS.add(task)

    def task_done(completed_task: asyncio.Task[None]) -> None:
        _NOTIFICATION_TASKS.discard(completed_task)
        _NOTIFICATION_QUEUE_SLOTS.release()

    task.add_done_callback(task_done)


def _dispatch_standalone_extension(
    callback_name: str,
    callback_args: tuple[Any, ...],
    label: str,
) -> None:
    """Run a callback from a standalone gRPC server without blocking it."""
    callback_slots = _NOTIFICATION_STANDALONE_CALLBACK_SLOTS
    if not callback_slots.acquire(  # pylint: disable=consider-using-with
        blocking=False
    ):
        log(WARNING, "%s extension notification queue is full.", label)
        return

    def invoke_callback() -> None:
        try:
            _invoke_extension(callback_name, callback_args, label)
        finally:
            callback_slots.release()

    callback_thread = threading.Thread(  # pylint: disable=consider-using-with
        target=invoke_callback,
        name="superlink-standalone-extension-callback",
        daemon=True,
    )
    try:
        callback_thread.start()
    except Exception:  # pylint: disable=broad-exception-caught
        callback_slots.release()
        log(WARNING, "%s extension notification could not start.", label)


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
        global _NOTIFICATION_PENDING_SUBMISSIONS  # pylint: disable=global-statement
        with _NOTIFICATION_PENDING_SUBMISSIONS_LOCK:
            _NOTIFICATION_PENDING_SUBMISSIONS += 1
        try:
            loop.call_soon_threadsafe(
                _schedule_extension, callback_name, callback_args, label
            )
            return
        except RuntimeError:
            # The service can begin shutdown between checking ``is_running`` and
            # scheduling. Fall back to the direct path so notifications are not
            # silently lost during orderly shutdown.
            with _NOTIFICATION_PENDING_SUBMISSIONS_LOCK:
                _NOTIFICATION_PENDING_SUBMISSIONS -= 1
            _NOTIFICATION_QUEUE_SLOTS.release()

    # Standalone gRPC handlers do not own the combined FastAPI lifespan. Keep
    # their completion callbacks non-blocking with the same bounded worker
    # policy used by the combined service.
    _dispatch_standalone_extension(callback_name, callback_args, label)


def notify_run_started(run: Run, source: RunStartSource) -> None:
    """Notify an optional extension after a run has been created successfully.

    In the combined SuperLink service the callback is scheduled on the existing
    FastAPI event loop, so it does not delay the gRPC or HTTP response. It
    receives a snapshot so it cannot mutate the run stored by SuperLink.
    Optional extension failures are logged and ignored. Standalone callers
    without a configured loop must provide their own non-blocking boundary.
    """
    _notify_extension("on_run_started", (run, source), "Run-started")
