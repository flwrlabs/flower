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
"""Private model task process helpers."""


import sys
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .run_model import run_model

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "run_model": ("flwr.supercore.task_process.model.run_model", "run_model"),
}

__all__ = ["run_model"]


class _ModelModule(ModuleType):
    """Preserve callable exports when their implementation modules are imported."""

    def __setattr__(self, name: str, value: object) -> None:
        """Avoid child modules shadowing same-named callable exports."""
        if name in _LAZY_EXPORTS:
            module_name, _ = _LAZY_EXPORTS[name]
            if isinstance(value, ModuleType) and value.__name__ == module_name:
                return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _ModelModule


def __getattr__(name: str) -> Any:
    """Lazily resolve the model-process entrypoint."""
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return eager and lazy Model-process entrypoints for completion."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
