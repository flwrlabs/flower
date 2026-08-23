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
# gRPC or HTTP), so only internally scheduled automations have an authoritative
# source. All API-created runs are reported as unknown.
RunStartSource = Literal["automation", "unknown"]
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


def _notify_extension(
    callback_name: str,
    callback_args: tuple[Any, ...],
    label: str,
) -> None:
    """Invoke one optional extension callback without affecting the request."""
    try:
        sgxt = _try_import_sgxt()
        if sgxt is None:
            return
        callback = cast(Callable[..., None] | None, getattr(sgxt, callback_name, None))
        if callback is not None:
            callback(*callback_args)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        log(
            WARNING,
            "%s extension notification failed: %s.",
            label,
            type(exc).__name__,
            exc_info=exc,
        )


def notify_run_started(run: Run, source: RunStartSource) -> None:
    """Notify an optional extension after a run has been created successfully.

    The callback is intentionally synchronous and must only perform bounded,
    non-blocking work. It receives a snapshot so it cannot mutate the run stored
    by SuperLink. Optional extension failures are logged and ignored.
    """
    _notify_extension(
        "on_run_started",
        (deepcopy(run), source),
        "Run-started",
    )
