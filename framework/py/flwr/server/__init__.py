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
"""Flower server."""


from importlib import import_module
from typing import TYPE_CHECKING, Any

from ..compat.server.app import start_server as start_server  # Deprecated
from . import strategy
from . import workflow as workflow
from .client_manager import ClientManager as ClientManager
from .client_manager import SimpleClientManager as SimpleClientManager
from .compat import LegacyContext as LegacyContext
from .history import History as History
from .server import Server as Server
from .server_config import ServerConfig as ServerConfig
from .serverapp_components import ServerAppComponents as ServerAppComponents

if TYPE_CHECKING:
    from flwr.compat.server.grid import Driver as Driver
    from flwr.serverapp import Grid as Grid
    from flwr.serverapp import ServerApp as ServerApp

_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "Driver": ("flwr.compat.server.grid", "Driver"),
    "Grid": ("flwr.serverapp", "Grid"),
    "ServerApp": ("flwr.serverapp", "ServerApp"),
}


def __getattr__(name: str) -> Any:
    """Lazily resolve compatibility exports."""
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name)
        value = module if attr_name is None else getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ClientManager",
    "Driver",
    "Grid",
    "History",
    "LegacyContext",
    "Server",
    "ServerApp",
    "ServerAppComponents",
    "ServerConfig",
    "SimpleClientManager",
    "start_server",
    "strategy",
    "workflow",
]
