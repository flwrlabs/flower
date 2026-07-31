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
from importlib import import_module
from types import ModuleType
from typing import Any, cast

from fastapi import FastAPI
from starlette.middleware import Middleware

SuperLinkLifespanContext = Callable[
    [FastAPI], AbstractAsyncContextManager[Mapping[str, Any] | None]
]
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
