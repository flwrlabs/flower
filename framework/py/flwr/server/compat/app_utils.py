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
"""Utility functions for the `start_grid`."""


import threading
from collections.abc import Callable
from typing import Any

from flwr.common.typing import RunNotRunningException
from flwr.serverapp.grid import Grid

from ..client_manager import ClientManager
from .grid_client_proxy import GridClientProxy


def start_update_client_manager_thread(
    grid: Grid,
    client_manager: ClientManager,
) -> tuple[threading.Thread, threading.Event, threading.Event]:
    """Periodically update the nodes list in the client manager in a thread.

    This function starts a thread that periodically uses the associated grid to
    get all node_ids. Each node_id is then converted into a `GridClientProxy`
    instance and stored in the `registered_nodes` dictionary with node_id as key.

    New nodes will be added to the ClientManager via `client_manager.register()`,
    and dead nodes will be removed from the ClientManager via
    `client_manager.unregister()`.

    Parameters
    ----------
    grid : Grid
        The Grid object to use.
    client_manager : ClientManager
        The ClientManager object to be updated.

    Returns
    -------
    threading.Thread
        A thread that updates the ClientManager and handles the stop event.
    threading.Event
        An event that, when set, signals the thread to stop.
    threading.Event
        An event that, when set, signals the node registration done.
    """
    f_stop = threading.Event()
    c_done = threading.Event()
    thread = threading.Thread(
        target=_update_client_manager,
        args=(
            grid,
            client_manager,
            f_stop,
            c_done,
        ),
        daemon=True,
    )
    thread.start()

    return thread, f_stop, c_done


def _update_client_manager(
    grid: Grid,
    client_manager: ClientManager,
    f_stop: threading.Event,
    c_done: threading.Event,
) -> None:
    """Update the nodes list in the client manager."""
    # Loop until the grid is disconnected
    registered_nodes: dict[int, GridClientProxy] = {}
    lock = threading.RLock()

    def update_registered_nodes() -> None:
        with lock:
            all_node_ids = set(grid.get_node_ids())
            dead_nodes = set(registered_nodes).difference(all_node_ids)
            new_nodes = all_node_ids.difference(registered_nodes)

            # Unregister dead nodes
            for node_id in dead_nodes:
                client_proxy = registered_nodes[node_id]
                client_manager.unregister(client_proxy)
                del registered_nodes[node_id]

            # Register new nodes
            for node_id in new_nodes:
                client_proxy = GridClientProxy(
                    node_id=node_id,
                    grid=grid,
                    run_id=grid.run.run_id,
                )
                if client_manager.register(client_proxy):
                    registered_nodes[node_id] = client_proxy
                else:
                    raise RuntimeError("Could not register node.")

    # Get the wrapped method of ClientManager instance
    def get_wrapped_method(method_name: str) -> Callable[..., Any]:
        original_method = getattr(client_manager, method_name)

        def wrapped_method(*args: Any, **kwargs: Any) -> Any:
            # Update registered nodes before calling the original method
            update_registered_nodes()
            return original_method(*args, **kwargs)

        return wrapped_method

    # Wrap the ClientManager
    for method_name in ["num_available", "all", "sample"]:
        setattr(client_manager, method_name, get_wrapped_method(method_name))

    c_done.set()

    while not f_stop.is_set():
        # Sleep for 5 seconds
        if not f_stop.is_set():
            f_stop.wait(5)

        try:
            # Update registered nodes
            update_registered_nodes()
        except RunNotRunningException:
            f_stop.set()
            break
