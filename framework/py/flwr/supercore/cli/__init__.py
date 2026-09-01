# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Flower command line interface for shared infrastructure components."""


from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .flower_superexec import flower_superexec
    from .flwr_agentapp import flwr_agentapp
    from .flwr_connector import flwr_connector
    from .flwr_model import flwr_model

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "flower_superexec": ("flwr.supercore.cli.flower_superexec", "flower_superexec"),
    "flwr_agentapp": ("flwr.supercore.cli.flwr_agentapp", "flwr_agentapp"),
    "flwr_connector": ("flwr.supercore.cli.flwr_connector", "flwr_connector"),
    "flwr_model": ("flwr.supercore.cli.flwr_model", "flwr_model"),
}

__all__ = [
    "flower_superexec",
    "flwr_agentapp",
    "flwr_connector",
    "flwr_model",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve console entrypoint functions."""
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return eager and lazy entrypoints for interactive completion."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
