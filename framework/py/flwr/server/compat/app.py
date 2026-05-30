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
"""Flower grid app."""


from logging import INFO

from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.server import Server, init_defaults, run_fl
from flwr.server.server_config import ServerConfig
from flwr.server.strategy import Strategy
from flwr.serverapp.grid import Grid

from .app_utils import start_update_client_manager_thread


def start_grid(  # pylint: disable=too-many-arguments, too-many-locals
    *,
    grid: Grid,
    server: Server | None = None,
    config: ServerConfig | None = None,
    strategy: Strategy | None = None,
    client_manager: ClientManager | None = None,
) -> History:
    """Start a Flower server.

    Parameters
    ----------
    grid : Grid
        The Grid object to use.
    server : Optional[flwr.server.Server] (default: None)
        A server implementation, either `flwr.server.Server` or a subclass
        thereof. If no instance is provided, then `start_grid` will create
        one.
    config : Optional[ServerConfig] (default: None)
        Currently supported values are `num_rounds` (int, default: 1) and
        `round_timeout` in seconds (float, default: None).
    strategy : Optional[flwr.server.Strategy] (default: None).
        An implementation of the abstract base class
        `flwr.server.strategy.Strategy`. If no strategy is provided, then
        `start_server` will use `flwr.server.strategy.FedAvg`.
    client_manager : Optional[flwr.server.ClientManager] (default: None)
        An implementation of the class `flwr.server.ClientManager`. If no
        implementation is provided, then `start_grid` will use
        `flwr.server.SimpleClientManager`.

    Returns
    -------
    hist : flwr.server.history.History
        Object containing training and evaluation metrics.
    """
    # Initialize the server and config
    initialized_server, initialized_config = init_defaults(
        server=server,
        config=config,
        strategy=strategy,
        client_manager=client_manager,
    )
    log(
        INFO,
        "Starting Flower ServerApp, config: %s",
        initialized_config,
    )
    log(INFO, "")

    # Start the thread updating nodes
    thread, f_stop, c_done = start_update_client_manager_thread(
        grid, initialized_server.client_manager()
    )

    # Wait until the node registration done
    c_done.wait()

    # Start training
    hist = run_fl(
        server=initialized_server,
        config=initialized_config,
    )

    # Terminate the thread
    f_stop.set()
    thread.join()

    return hist
