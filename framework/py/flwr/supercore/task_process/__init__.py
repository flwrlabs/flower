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
"""Flower task process components."""


from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent.run_agentapp import run_agentapp
    from .connector.run_connector import run_connector
    from .model.run_model import run_model

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "run_agentapp": ("flwr.supercore.task_process.agent", "run_agentapp"),
    "run_connector": ("flwr.supercore.task_process.connector", "run_connector"),
    "run_model": ("flwr.supercore.task_process.model", "run_model"),
}

__all__ = [
    "run_agentapp",
    "run_connector",
    "run_model",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve task-process entrypoint functions."""
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return eager and lazy task-process entrypoints for completion."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
