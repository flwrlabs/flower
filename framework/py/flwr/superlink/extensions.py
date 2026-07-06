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
from typing import Any

from fastapi import FastAPI

SuperLinkLifespanContext = Callable[
    [FastAPI], AbstractAsyncContextManager[Mapping[str, Any] | None]
]


def configure_app(app: FastAPI) -> None:
    """Configure SuperLink FastAPI extensions."""
    try:
        # pylint: disable-next=import-outside-toplevel
        from flwr.ee.superlink.extensions import configure_app as _configure_ee_app
    except ModuleNotFoundError:
        return

    configure_ee_app: Callable[[FastAPI], None]
    configure_ee_app = _configure_ee_app
    configure_ee_app(app)


def get_lifespan_contexts() -> tuple[SuperLinkLifespanContext, ...]:
    """Return SuperLink FastAPI lifespan contexts."""
    try:
        # pylint: disable-next=import-outside-toplevel
        from flwr.ee.superlink.extensions import (
            get_lifespan_contexts as _get_ee_lifespan_contexts,
        )
    except ModuleNotFoundError:
        return ()

    get_ee_lifespan_contexts: Callable[[], tuple[SuperLinkLifespanContext, ...]]
    get_ee_lifespan_contexts = _get_ee_lifespan_contexts
    return get_ee_lifespan_contexts()
