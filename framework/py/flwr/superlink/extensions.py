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


def configure_app(_: FastAPI) -> None:
    """Configure SuperLink FastAPI extensions."""


def get_lifespan_contexts() -> tuple[SuperLinkLifespanContext, ...]:
    """Return SuperLink FastAPI lifespan contexts."""
    return ()
