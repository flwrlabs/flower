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
"""Flower simulation app."""


import asyncio
import logging
import sys
import threading
import traceback
import warnings
from logging import ERROR, INFO
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from flwr.client import ClientFnExt
from flwr.common import EventType, event
from flwr.common.constant import NODE_ID_NUM_BYTES, SUPERLINK_NODE_ID
from flwr.common.logger import (
    log,
    set_logger_propagation,
    warn_deprecated_feature,
    warn_unsupported_feature,
)
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.server import Server, init_defaults, run_fl
from flwr.server.server_config import ServerConfig
from flwr.server.strategy import Strategy
from flwr.server.superlink.linkstate.utils import generate_rand_int_from_bytes
from flwr.simulation.ray_transport.ray_actor import (
    ClientAppActor,
    VirtualClientEngineActor,
    VirtualClientEngineActorPool,
    pool_size_from_resources,
)
from flwr.simulation.ray_transport.ray_client_proxy import RayActorClientProxy

INVALID_ARGUMENTS_START_SIMULATION = """
INVALID ARGUMENTS ERROR

Invalid Arguments in method:

`start_simulation(
    *,
    client_fn: ClientFn,
    num_clients: int,
    clients_ids: Optional[List[str]] = None,
    client_resources: Optional[Dict[str, float]] = None,
    server: Optional[Server] = None,
    config: ServerConfig = None,
    strategy: Optional[Strategy] = None,
    client_manager: Optional[ClientManager] = None,
    ray_init_args: Optional[Dict[str, Any]] = None,
) -> None:`

REASON:
    Method requires:
        - Either `num_clients`[int] or `clients_ids`[List[str]]
        to be set exclusively.
        OR
        - `len(clients_ids)` == `num_clients`

"""

NodeToPartitionMapping = dict[int, int]


def _create_node_id_to_partition_mapping(
    num_clients: int,
) -> NodeToPartitionMapping:
    """Generate a node_id:partition_id mapping."""
    nodes_mapping: NodeToPartitionMapping = {}  # {node-id; partition-id}
    for i in range(num_clients):
        while True:
            node_id = generate_rand_int_from_bytes(
                NODE_ID_NUM_BYTES, exclude={SUPERLINK_NODE_ID, 0}
            )
            if node_id not in nodes_mapping:
                break
        nodes_mapping[node_id] = i
    return nodes_mapping


# pylint: disable=too-many-arguments,too-many-statements,too-many-branches
def start_simulation(
    *,
    client_fn: ClientFnExt,
    num_clients: int,
    clients_ids: list[str] | None = None,  # UNSUPPORTED, WILL BE REMOVED
    client_resources: dict[str, float] | None = None,
    server: Server | None = None,
    config: ServerConfig | None = None,
    strategy: Strategy | None = None,
    client_manager: ClientManager | None = None,
    ray_init_args: dict[str, Any] | None = None,
    keep_initialised: bool | None = False,
    actor_type: type[VirtualClientEngineActor] = ClientAppActor,
    actor_kwargs: dict[str, Any] | None = None,
    actor_scheduling: str | NodeAffinitySchedulingStrategy = "DEFAULT",
) -> History:
    """Start a Ray-based Flower simulation server.

    Warning
    -------
    This function is deprecated since 1.13.0. Use :code: `flwr run` to start a Flower
    simulation.

    Parameters
    ----------
    client_fn : ClientFnExt
        A function creating `Client` instances. The function must have the signature
        `client_fn(context: Context). It should return
        a single client instance of type `Client`. Note that the created client
        instances are ephemeral and will often be destroyed after a single method
        invocation. Since client instances are not long-lived, they should not attempt
        to carry state over method invocations. Any state required by the instance
        (model, dataset, hyperparameters, ...) should be (re-)created in either the
        call to `client_fn` or the call to any of the client methods (e.g., load
        evaluation data in the `evaluate` method itself).
    num_clients : int
        The total number of clients in this simulation.
    clients_ids : Optional[List[str]]
        UNSUPPORTED, WILL BE REMOVED. USE `num_clients` INSTEAD.
        List `client_id`s for each client. This is only required if
        `num_clients` is not set. Setting both `num_clients` and `clients_ids`
        with `len(clients_ids)` not equal to `num_clients` generates an error.
        Using this argument will raise an error.
    client_resources : Optional[Dict[str, float]] (default: `{"num_cpus": 1, "num_gpus": 0.0}`)
        CPU and GPU resources for a single client. Supported keys
        are `num_cpus` and `num_gpus`. To understand the GPU utilization caused by
        `num_gpus`, as well as using custom resources, please consult the Ray
        documentation.
    server : Optional[flwr.server.Server] (default: None).
        An implementation of the abstract base class `flwr.server.Server`. If no
        instance is provided, then `start_server` will create one.
    config: ServerConfig (default: None).
        Currently supported values are `num_rounds` (int, default: 1) and
        `round_timeout` in seconds (float, default: None).
    strategy : Optional[flwr.server.Strategy] (default: None)
        An implementation of the abstract base class `flwr.server.Strategy`. If
        no strategy is provided, then `start_server` will use
        `flwr.server.strategy.FedAvg`.
    client_manager : Optional[flwr.server.ClientManager] (default: None)
        An implementation of the abstract base class `flwr.server.ClientManager`.
        If no implementation is provided, then `start_simulation` will use
        `flwr.server.client_manager.SimpleClientManager`.
    ray_init_args : Optional[Dict[str, Any]] (default: None)
        Optional dictionary containing arguments for the call to `ray.init`.
        If ray_init_args is None (the default), Ray will be initialized with
        the following default args:

        { "ignore_reinit_error": True, "include_dashboard": False }

        An empty dictionary can be used (ray_init_args={}) to prevent any
        arguments from being passed to ray.init.
    keep_initialised: Optional[bool] (default: False)
        Set to True to prevent `ray.shutdown()` in case `ray.is_initialized()=True`.

    actor_type: VirtualClientEngineActor (default: ClientAppActor)
        Optionally specify the type of actor to use. The actor object, which
        persists throughout the simulation, will be the process in charge of
        executing a ClientApp wrapping input argument `client_fn`.

    actor_kwargs: Optional[Dict[str, Any]] (default: None)
        If you want to create your own Actor classes, you might need to pass
        some input argument. You can use this dictionary for such purpose.

    actor_scheduling: Optional[Union[str, NodeAffinitySchedulingStrategy]]
        (default: "DEFAULT")
        Optional string ("DEFAULT" or "SPREAD") for the VCE to choose in which
        node the actor is placed. If you are an advanced user needed more control
        you can use lower-level scheduling strategies to pin actors to specific
        compute nodes (e.g. via NodeAffinitySchedulingStrategy). Please note this
        is an advanced feature. For all details, please refer to the Ray documentation:
        https://docs.ray.io/en/latest/ray-core/scheduling/index.html

    Returns
    -------
    hist : flwr.server.history.History
        Object containing metrics from training.
    """  # noqa: E501
    # pylint: disable-msg=too-many-locals
    msg = (
        "flwr.simulation.start_simulation() is deprecated."
        "\n\tInstead, use the `flwr run` CLI command to start a local simulation "
        "in your Flower app, as shown for example below:"
        "\n\n\t\t$ flwr new  # Create a new Flower app from a template"
        "\n\n\t\t$ flwr run  # Run the Flower app in Simulation Mode"
        "\n\n\tUsing `start_simulation()` is deprecated."
    )
    warn_deprecated_feature(name=msg)

    event(
        EventType.START_SIMULATION_ENTER,
        {"num_clients": len(clients_ids) if clients_ids is not None else num_clients},
    )

    if clients_ids is not None:
        warn_unsupported_feature(
            "Passing `clients_ids` to `start_simulation` is deprecated and not longer "
            "used by `start_simulation`. Use `num_clients` exclusively instead."
        )
        log(ERROR, "`clients_ids` argument used.")
        sys.exit()

    # Set logger propagation
    loop: asyncio.AbstractEventLoop | None = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    finally:
        if loop and loop.is_running():
            # Set logger propagation to False to prevent duplicated log output in Colab.
            logger = logging.getLogger("flwr")
            _ = set_logger_propagation(logger, False)

    # Initialize server and server config
    initialized_server, initialized_config = init_defaults(
        server=server,
        config=config,
        strategy=strategy,
        client_manager=client_manager,
    )

    log(
        INFO,
        "Starting Flower simulation, config: %s",
        initialized_config,
    )

    # Create node-id to partition-id mapping
    nodes_mapping = _create_node_id_to_partition_mapping(num_clients)

    # Default arguments for Ray initialization
    if not ray_init_args:
        ray_init_args = {
            "ignore_reinit_error": True,
            "include_dashboard": False,
        }

    # Shut down Ray if it has already been initialized (unless asked not to)
    if ray.is_initialized() and not keep_initialised:
        ray.shutdown()

    # Initialize Ray
    ray.init(**ray_init_args)
    cluster_resources = ray.cluster_resources()
    log(
        INFO,
        "Flower VCE: Ray initialized with resources: %s",
        cluster_resources,
    )

    log(
        INFO,
        "Optimize your simulation with Flower VCE: "
        "https://flower.ai/docs/framework/how-to-run-simulations.html",
    )

    # Log the resources that a single client will be able to use
    if client_resources is None:
        log(
            INFO,
            "No `client_resources` specified. Using minimal resources for clients.",
        )
        client_resources = {"num_cpus": 1, "num_gpus": 0.0}

    # Each client needs at the very least one CPU
    if "num_cpus" not in client_resources:
        warnings.warn(
            "No `num_cpus` specified in `client_resources`. "
            "Using `num_cpus=1` for each client.",
            stacklevel=2,
        )
        client_resources["num_cpus"] = 1

    log(
        INFO,
        "Flower VCE: Resources for each Virtual Client: %s",
        client_resources,
    )

    actor_args = {} if actor_kwargs is None else actor_kwargs

    # An actor factory. This is called N times to add N actors
    # to the pool. If at some point the pool can accommodate more actors
    # this will be called again.
    def create_actor_fn() -> type[VirtualClientEngineActor]:
        return actor_type.options(  # type: ignore
            **client_resources,
            scheduling_strategy=actor_scheduling,
        ).remote(**actor_args)

    # Instantiate ActorPool
    pool = VirtualClientEngineActorPool(
        create_actor_fn=create_actor_fn,
        client_resources=client_resources,
    )

    f_stop = threading.Event()

    # Periodically, check if the cluster has grown (i.e. a new
    # node has been added). If this happens, we likely want to grow
    # the actor pool by adding more Actors to it.
    def update_resources(f_stop: threading.Event) -> None:
        """Periodically check if more actors can be added to the pool.

        If so, extend the pool.
        """
        if not f_stop.is_set():
            num_max_actors = pool_size_from_resources(client_resources)
            if num_max_actors > pool.num_actors:
                num_new = num_max_actors - pool.num_actors
                log(
                    INFO, "The cluster expanded. Adding %s actors to the pool.", num_new
                )
                pool.add_actors_to_pool(num_actors=num_new)

            threading.Timer(10, update_resources, [f_stop]).start()

    update_resources(f_stop)

    log(
        INFO,
        "Flower VCE: Creating %s with %s actors",
        pool.__class__.__name__,
        pool.num_actors,
    )

    # Register one RayClientProxy object for each client with the ClientManager
    for node_id, partition_id in nodes_mapping.items():
        client_proxy = RayActorClientProxy(
            client_fn=client_fn,
            node_id=node_id,
            partition_id=partition_id,
            num_partitions=num_clients,
            actor_pool=pool,
        )
        initialized_server.client_manager().register(client=client_proxy)

    hist = History()
    # pylint: disable=broad-except
    try:
        # Start training
        hist = run_fl(
            server=initialized_server,
            config=initialized_config,
        )
    except Exception as ex:
        log(ERROR, ex)
        log(ERROR, traceback.format_exc())
        log(
            ERROR,
            "Your simulation crashed :(. This could be because of several reasons. "
            "The most common are: "
            "\n\t > Sometimes, issues in the simulation code itself can cause crashes. "
            "It's always a good idea to double-check your code for any potential bugs "
            "or inconsistencies that might be contributing to the problem. "
            "For example: "
            "\n\t\t - You might be using a class attribute in your clients that "
            "hasn't been defined."
            "\n\t\t - There could be an incorrect method call to a 3rd party library "
            "(e.g., PyTorch)."
            "\n\t\t - The return types of methods in your clients/strategies might be "
            "incorrect."
            "\n\t > Your system couldn't fit a single VirtualClient: try lowering "
            "`client_resources`."
            "\n\t > All the actors in your pool crashed. This could be because: "
            "\n\t\t - You clients hit an out-of-memory (OOM) error and actors couldn't "
            "recover from it. Try launching your simulation with more generous "
            "`client_resources` setting (i.e. it seems %s is "
            "not enough for your run). Use fewer concurrent actors. "
            "\n\t\t - You were running a multi-node simulation and all worker nodes "
            "disconnected. The head node might still be alive but cannot accommodate "
            "any actor with resources: %s."
            "\nTake a look at the Flower simulation examples for guidance "
            "<https://flower.ai/docs/framework/how-to-run-simulations.html>.",
            client_resources,
            client_resources,
        )
        raise RuntimeError("Simulation crashed.") from ex

    finally:
        # Stop time monitoring resources in cluster
        f_stop.set()
        event(EventType.START_SIMULATION_LEAVE)

    return hist
