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
from typing import Any, Protocol, cast

from fastapi import FastAPI

EE_EXTENSIONS_MODULE = "flwr.ee.superlink.extensions"

SuperLinkLifespanContext = Callable[
    [FastAPI], AbstractAsyncContextManager[Mapping[str, Any] | None]
]


class _SuperLinkExtensions(Protocol):
    def configure_app(self, app: FastAPI) -> None:
        """Configure the FastAPI app."""
        pass

    def get_lifespan_contexts(self) -> tuple[SuperLinkLifespanContext, ...]:
        """Return lifespan contexts."""
        pass


def _get_ee_extensions() -> _SuperLinkExtensions | None:
    try:
        module = import_module(EE_EXTENSIONS_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name in {"flwr.ee", "flwr.ee.superlink", EE_EXTENSIONS_MODULE}:
            return None
        raise

    return cast(_SuperLinkExtensions, module)


def configure_app(app: FastAPI) -> None:
    """Configure SuperLink FastAPI extensions."""
    ee_extensions = _get_ee_extensions()
    if ee_extensions is None:
        return

    ee_extensions.configure_app(app)


def get_lifespan_contexts() -> tuple[SuperLinkLifespanContext, ...]:
    """Return SuperLink FastAPI lifespan contexts."""
    ee_extensions = _get_ee_extensions()
    if ee_extensions is None:
        return ()

    return ee_extensions.get_lifespan_contexts()
