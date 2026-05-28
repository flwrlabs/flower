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
"""JSON type aliases for AgentApp APIs."""


from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias

from flwr.common import Context

if TYPE_CHECKING:
    from .session import AgentSession

AgentAppCallable: TypeAlias = Callable[["AgentSession", Context], "JSONObject"]

JSONObject: TypeAlias = "dict[str, JSONValue]"
JSONValue: TypeAlias = "None | bool | int | float | str | list[JSONValue] | JSONObject"

__all__ = ["AgentAppCallable", "JSONObject", "JSONValue"]
