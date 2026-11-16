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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import agentapp as agentapp
    from . import app as app
    from . import client as client
    from . import clientapp as clientapp
    from . import common as common
    from . import server as server
    from . import serverapp as serverapp

    __version__: str


__all__ = [
    "agentapp",
    "app",
    "clientapp",
    "serverapp",
]

_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "__version__": ("flwr.supercore.version", "package_version"),
    "agentapp": ("flwr.agentapp", None),
    "app": ("flwr.app", None),
    "client": ("flwr.client", None),
    "clientapp": ("flwr.clientapp", None),
    "common": ("flwr.common", None),
    "server": ("flwr.server", None),
    "serverapp": ("flwr.serverapp", None),
    "simulation": ("flwr.simulation", None),
}


def __getattr__(name: str) -> Any:
    """Import package exports only when requested."""
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name)
        value = module if attr_name is None else getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Keep lazy exports visible to package introspection."""
    return sorted(set(globals()) | _LAZY_EXPORTS.keys())
