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


def __getattr__(name: str) -> Any:
    """Lazily resolve compatibility exports."""
    if name == "Driver":
        # pylint: disable=import-outside-toplevel
        from flwr.compat.server.grid import Driver

        # pylint: enable=import-outside-toplevel
        globals()[name] = Driver
        return Driver
    if name == "Grid":
        # pylint: disable=import-outside-toplevel
        from flwr.serverapp import Grid

        # pylint: enable=import-outside-toplevel
        globals()[name] = Grid
        return Grid
    if name == "ServerApp":
        from flwr.serverapp import ServerApp  # pylint: disable=import-outside-toplevel

        globals()[name] = ServerApp
        return ServerApp
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
