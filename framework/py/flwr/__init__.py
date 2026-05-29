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


import importlib

from flwr.supercore.version import package_version as _package_version

from . import agentapp, app, clientapp, serverapp

__all__ = [
    "agentapp",
    "app",
    "clientapp",
    "serverapp",
]

__version__ = _package_version


# Lazy imports for legacy support
_lazy_imports = {"simulation", "server", "client", "common"}


def __getattr__(name: str) -> object:
    """Lazy import for legacy support."""
    if name in _lazy_imports:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
