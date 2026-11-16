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
"""Legacy ServerAppComponents for the ServerApp."""


from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flwr.server.client_manager import ClientManager
    from flwr.server.server import Server
    from flwr.server.server_config import ServerConfig
    from flwr.server.strategy import Strategy


@dataclass
class ServerAppComponents:  # pylint: disable=too-many-instance-attributes
    """Components to construct a ServerApp.

    Parameters
    ----------
    server : Optional[Server] (default: None)
        A server implementation, either `flwr.server.Server` or a subclass
        thereof. If no instance is provided, one will be created internally.
    config : Optional[ServerConfig] (default: None)
        Currently supported values are `num_rounds` (int, default: 1) and
        `round_timeout` in seconds (float, default: None).
    strategy : Optional[Strategy] (default: None)
        An implementation of the abstract base class
        `flwr.server.strategy.Strategy`. If no strategy is provided, then
        `flwr.server.strategy.FedAvg` will be used.
    client_manager : Optional[ClientManager] (default: None)
        An implementation of the class `flwr.server.ClientManager`. If no
        implementation is provided, then `flwr.server.SimpleClientManager`
        will be used.
    """

    server: Server | None = None
    config: ServerConfig | None = None
    strategy: Strategy | None = None
    client_manager: ClientManager | None = None
