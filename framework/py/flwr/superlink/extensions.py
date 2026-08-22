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


import queue
import threading
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
RUN_START_SOURCE_METADATA_KEY = "x-flwr-run-source"
RunStartSource = Literal["grpc", "http", "web_ui", "automation", "unknown"]
_RUN_START_SOURCES = frozenset({"grpc", "http", "web_ui", "automation", "unknown"})
_SGXT_MODULE = "flwr.ee.superlink.extensions"
_RUN_STARTED_NOTIFICATION_CAPACITY = 1000
_RUN_STARTED_CALLBACK_CAPACITY = 4
_RUN_STARTED_CALLBACK_TIMEOUT_SECONDS = 5.0
_RUN_STARTED_NOTIFICATIONS: queue.Queue[tuple[Run, RunStartSource]] = queue.Queue(
    maxsize=_RUN_STARTED_NOTIFICATION_CAPACITY
)
_RUN_STARTED_CALLBACK_SLOTS = threading.BoundedSemaphore(_RUN_STARTED_CALLBACK_CAPACITY)


def resolve_run_start_source(
    value: str | bytes | None, *, default: RunStartSource
) -> RunStartSource:
    """Resolve an optional caller-provided run source to a closed value set."""
    if value is None:
        return default
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            return "unknown"
    if value not in _RUN_START_SOURCES:
        return "unknown"
    return cast(RunStartSource, value)


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


def _invoke_run_started(  # pylint: disable=consider-using-with
    run: Run,
    source: RunStartSource,
) -> None:
    """Discover and invoke one extension outside the run-creation request path."""
    if not _RUN_STARTED_CALLBACK_SLOTS.acquire(blocking=False):
        log(WARNING, "Run-started extension callback capacity is exhausted.")
        return

    completed = threading.Event()

    def invoke_callback() -> None:
        try:
            sgxt = _try_import_sgxt()
            if sgxt is None:
                return
            callback = cast(
                Callable[[Run, RunStartSource], None] | None,
                getattr(sgxt, "on_run_started", None),
            )
            if callback is not None:
                # Extensions receive a snapshot so they cannot mutate persisted state.
                callback(deepcopy(run), source)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            log(
                WARNING,
                "Run-started extension notification failed: %s.",
                type(exc).__name__,
                exc_info=exc,
            )
        finally:
            _RUN_STARTED_CALLBACK_SLOTS.release()
            completed.set()

    callback_thread = threading.Thread(
        target=invoke_callback,
        name="run-started-extension-callback",
        daemon=True,
    )
    try:
        callback_thread.start()
    except Exception:  # pylint: disable=broad-exception-caught
        _RUN_STARTED_CALLBACK_SLOTS.release()
        raise
    if not completed.wait(timeout=_RUN_STARTED_CALLBACK_TIMEOUT_SECONDS):
        log(WARNING, "Run-started extension notification timed out.")


def _run_started_dispatcher() -> None:
    """Dispatch bounded notifications on a daemon worker until process exit."""
    while True:
        run, source = _RUN_STARTED_NOTIFICATIONS.get()
        try:
            _invoke_run_started(run, source)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            log(
                WARNING,
                "Run-started extension dispatch failed: %s.",
                type(exc).__name__,
                exc_info=exc,
            )
        finally:
            _RUN_STARTED_NOTIFICATIONS.task_done()


_RUN_STARTED_DISPATCHER = threading.Thread(
    target=_run_started_dispatcher,
    name="run-started-extension-dispatcher",
    daemon=True,
)
_RUN_STARTED_DISPATCHER.start()


def notify_run_started(run: Run, source: RunStartSource) -> None:
    """Schedule a bounded notification after a run was created successfully."""
    try:
        _RUN_STARTED_NOTIFICATIONS.put_nowait((run, source))
    except queue.Full:
        log(WARNING, "Run-started extension notification queue is full.")
