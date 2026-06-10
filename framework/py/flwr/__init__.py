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
"""Flower main package."""


from importlib import import_module
from typing import Any

from flwr.supercore.version import package_version as _package_version

from . import agentapp, app
from . import client as client
from . import clientapp
from . import common as common
from . import server as server
from . import serverapp

__all__ = [
    "agentapp",
    "app",
    "clientapp",
    "serverapp",
]

__version__ = _package_version


_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "simulation": ("flwr.simulation", None),
}


def __getattr__(name: str) -> Any:
    """Lazy import for legacy support."""
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name)
        value = module if attr_name is None else getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
