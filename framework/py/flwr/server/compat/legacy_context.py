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
"""Legacy Context."""


from dataclasses import dataclass

from flwr.app import Context

from ..client_manager import ClientManager, SimpleClientManager
from ..history import History
from ..server_config import ServerConfig
from ..strategy import FedAvg, Strategy


@dataclass
class LegacyContext(Context):
    """Legacy Context."""

    config: ServerConfig
    strategy: Strategy
    client_manager: ClientManager
    history: History

    def __init__(  # pylint: disable=too-many-arguments
        self,
        context: Context,
        config: ServerConfig | None = None,
        strategy: Strategy | None = None,
        client_manager: ClientManager | None = None,
    ) -> None:
        if config is None:
            config = ServerConfig()
        if strategy is None:
            strategy = FedAvg()
        if client_manager is None:
            client_manager = SimpleClientManager()
        self.config = config
        self.strategy = strategy
        self.client_manager = client_manager
        self.history = History()

        super().__init__(**vars(context))
