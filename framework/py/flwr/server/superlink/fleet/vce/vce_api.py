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
"""Fleet Simulation Runtime API."""


import json
import secrets
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from logging import DEBUG, ERROR, INFO, WARN
from pathlib import Path
from queue import Empty, Queue
from uuid import uuid4

from flwr.app import Message
from flwr.app.error import Error
from flwr.clientapp.client_app import ClientApp, ClientAppException, LoadClientAppError
from flwr.clientapp.utils import get_load_client_app_fn
from flwr.common.constant import (
    HEARTBEAT_INTERVAL_INF,
    NOOP_ACCOUNT_NAME,
    NOOP_FLWR_AID,
    NUM_PARTITIONS_KEY,
    PARTITION_ID_KEY,
    ErrorCode,
)
from flwr.compat.client.run_info_store import DeprecatedRunInfoStore
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory
from flwr.supercore import log
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.run import Run
from flwr.superlink.federation import NoOpFederationManager

from .backend import Backend
from .metrics import VceMetrics

NodeToPartitionMapping = dict[int, int]


@dataclass
class PreparedVce:
    """Simulation runtime brought up before a seeded ServerApp starts."""

    app_fn: Callable[[], ClientApp]
    nodes_mapping: NodeToPartitionMapping
    state_factory: LinkStateFactory
    node_info_stores: dict[int, DeprecatedRunInfoStore]
    backend: Backend


def _register_nodes(
    num_nodes: int,
    state_factory: LinkStateFactory,
    random_seed: int | None = None,
) -> NodeToPartitionMapping:
    """Register nodes with the StateFactory and create node-id:partition-id mapping.

    Parameters
    ----------
    num_nodes : int
        Number of virtual nodes to register.
    state_factory : LinkStateFactory
        State factory that stores the nodes.
    random_seed : Optional[int] (default: None)
        When set, node ids are ``2 .. num_nodes + 1`` so client sampling selects
        the same partitions on every run. Id ``1`` is reserved for the SuperLink.
        The seed value itself does not change the ids; it only enables stable ids.
    """
    nodes_mapping: NodeToPartitionMapping = {}
    state = state_factory.state()
    for i in range(num_nodes):
        # No node authentication in simulation;
        # use NOOP_FLWR_AID as owner_aid and
        # use random bytes as public key.
        # Id 1 is the SuperLink, so seeded simulation nodes start at 2.
        explicit_node_id = None if random_seed is None else i + 2
        node_id = state.create_node(
            NOOP_FLWR_AID,
            NOOP_ACCOUNT_NAME,
            secrets.token_bytes(32),
            heartbeat_interval=HEARTBEAT_INTERVAL_INF,
            node_id=explicit_node_id,
        )
        state.acknowledge_node_heartbeat(
            node_id=node_id, heartbeat_interval=HEARTBEAT_INTERVAL_INF
        )
        nodes_mapping[node_id] = i
    log(DEBUG, "Registered %i nodes", len(nodes_mapping))
    return nodes_mapping


def _register_node_info_stores(
    nodes_mapping: NodeToPartitionMapping,
    run: Run,
    app_dir: str | None = None,
) -> dict[int, DeprecatedRunInfoStore]:
    """Create DeprecatedRunInfoStore objects and register the context for the run."""
    node_info_store: dict[int, DeprecatedRunInfoStore] = {}
    num_partitions = len(set(nodes_mapping.values()))
    for node_id, partition_id in nodes_mapping.items():
        node_info_store[node_id] = DeprecatedRunInfoStore(
            node_id=node_id,
            node_config={
                PARTITION_ID_KEY: partition_id,
                NUM_PARTITIONS_KEY: num_partitions,
            },
        )

        # Pre-register Context objects
        node_info_store[node_id].register_context(
            run_id=run.run_id, run=run, app_dir=app_dir
        )

    return node_info_store


# pylint: disable-next=too-many-arguments,too-many-locals,too-many-positional-arguments
def worker(
    messageins_queue: Queue[Message],
    messageres_queue: Queue[Message],
    node_info_store: dict[int, DeprecatedRunInfoStore],
    backend: Backend,
    f_stop: threading.Event,
    metrics: VceMetrics,
) -> None:
    """Process messages from the queue, execute them, update context, and enqueue
    replies."""
    while not f_stop.is_set():
        out_mssg = None
        processing_started_at = None
        try:
            # Fetch from queue with timeout. We use a timeout so
            # the stopping event can be evaluated even when the queue is empty.
            message: Message = messageins_queue.get(timeout=1.0)
            node_id = message.metadata.dst_node_id

            # Retrieve context
            context = node_info_store[node_id].retrieve_context(
                run_id=message.metadata.run_id
            )

            # Let backend process message
            processing_started_at = time.perf_counter()
            out_mssg, updated_context = backend.process_message(message, context)

            # Update Context
            node_info_store[node_id].update_context(
                message.metadata.run_id, context=updated_context
            )
        except Empty:
            # An exception raised if queue.get times out
            pass
        # Exceptions aren't raised but reported as an error message
        except Exception as ex:  # pylint: disable=broad-exception-caught
            log(ERROR, ex)
            log(ERROR, traceback.format_exc())

            if isinstance(ex, ClientAppException):
                e_code = ErrorCode.CLIENT_APP_RAISED_EXCEPTION
            elif isinstance(ex, LoadClientAppError):
                e_code = ErrorCode.LOAD_CLIENT_APP_EXCEPTION
            else:
                e_code = ErrorCode.UNKNOWN

            reason = str(type(ex)) + ":<'" + str(ex) + "'>"
            out_mssg = Message(Error(code=e_code, reason=reason), reply_to=message)

        finally:
            if processing_started_at is not None:
                metrics.add_clientapp_runtime(
                    time.perf_counter() - processing_started_at
                )
            if out_mssg:
                # Assign a message_id
                out_mssg.metadata.__dict__["_message_id"] = str(uuid4())
                # Store reply Messages in state
                messageres_queue.put(out_mssg)


def add_messages_to_queue(
    state: LinkState,
    queue: Queue[Message],
    nodes_mapping: NodeToPartitionMapping,
    f_stop: threading.Event,
) -> None:
    """Put Messages in the queue from the LinkState."""
    while not f_stop.is_set():
        for node_id in nodes_mapping.keys():
            messages = state.get_messages(dst_node_id=node_id, limit=1)
            for msg in messages:
                queue.put(msg)
                state.acknowledge_messages([msg.metadata.message_id])
        f_stop.wait(0.1)


def put_message_into_state(
    state: LinkState, queue: Queue[Message], f_stop: threading.Event
) -> None:
    """Store reply Messages into the LinkState from the queue."""
    while not f_stop.is_set():
        try:
            message_reply = queue.get(timeout=1.0)
            state.store_messages([message_reply])
        except Empty:
            # queue is empty when timeout was triggered
            pass


# pylint: disable-next=too-many-positional-arguments,too-many-arguments
def run_api(
    app_fn: Callable[[], ClientApp],
    backend_fn: Callable[[], Backend],
    nodes_mapping: NodeToPartitionMapping,
    state_factory: LinkStateFactory,
    node_info_stores: dict[int, DeprecatedRunInfoStore],
    f_stop: threading.Event,
    metrics: VceMetrics,
    backend: Backend | None = None,
) -> None:
    """Run the Simulation Runtime."""
    messageins_queue: Queue[Message] = Queue()
    messageres_queue: Queue[Message] = Queue()

    try:
        if backend is None:
            # Instantiate backend
            backend = backend_fn()

            # Build backend
            backend.build(app_fn)

        # Add workers (they submit Messages to Backend)
        state = state_factory.state()

        extractor_th = threading.Thread(
            target=add_messages_to_queue,
            args=(
                state,
                messageins_queue,
                nodes_mapping,
                f_stop,
            ),
        )
        extractor_th.start()

        injector_th = threading.Thread(
            target=put_message_into_state,
            args=(
                state,
                messageres_queue,
                f_stop,
            ),
        )
        injector_th.start()

        with ThreadPoolExecutor() as executor:
            _ = [
                executor.submit(
                    worker,
                    messageins_queue,
                    messageres_queue,
                    node_info_stores,
                    backend,
                    f_stop,
                    metrics,
                )
                for _ in range(backend.num_workers)
            ]

        extractor_th.join()
        injector_th.join()

    except Exception as ex:

        log(ERROR, "An exception occured!! %s", ex)
        log(ERROR, traceback.format_exc())
        log(WARN, "Stopping Simulation Runtime.")

        # Raise exception
        raise RuntimeError("Simulation Runtime crashed.") from ex

    finally:
        # Manually trigger stopping event
        f_stop.set()

        # Terminate backend
        if backend is not None:
            backend.terminate()


# pylint: disable=too-many-arguments,too-many-locals,too-many-branches
# pylint: disable=too-many-positional-arguments
def _vce_components(
    backend_config_json_stream: str,
    app_dir: str,
    is_app: bool,
    run: Run,
    client_app: ClientApp | None,
    client_app_attr: str | None,
    num_supernodes: int | None,
    state_factory: LinkStateFactory | None,
    existing_nodes_mapping: NodeToPartitionMapping | None,
    random_seed: int | None,
) -> tuple[
    Callable[[], ClientApp],
    Callable[[], Backend],
    NodeToPartitionMapping,
    LinkStateFactory,
    dict[int, DeprecatedRunInfoStore],
]:
    """Validate arguments and register nodes. Do not import the app or Ray."""
    nodes_mapping: NodeToPartitionMapping = {}

    if client_app_attr is not None and client_app is not None:
        raise ValueError(
            "Both `client_app_attr` and `client_app` are provided, "
            "but only one is allowed."
        )

    if num_supernodes is not None and existing_nodes_mapping is not None:
        raise ValueError(
            "Both `num_supernodes` and `existing_nodes_mapping` are provided, "
            "but only one is allowed."
        )
    if num_supernodes is None:
        if state_factory is None or existing_nodes_mapping is None:
            raise ValueError(
                "If not passing an existing `state_factory` and associated "
                "`existing_nodes_mapping` you must supply `num_supernodes` to indicate "
                "how many nodes to insert into a new StateFactory that will be created."
            )
    if existing_nodes_mapping:
        if state_factory is None:
            raise ValueError(
                "`existing_nodes_mapping` was passed, but no `state_factory` was "
                "passed."
            )
        log(INFO, "Using exiting NodeToPartitionMapping and StateFactory.")
        # Use mapping constructed externally. This also means nodes
        # have previously being registered.
        nodes_mapping = existing_nodes_mapping
    app_dir = str(Path(app_dir).absolute())

    if not state_factory:
        log(INFO, "A StateFactory was not supplied to the SimulationEngine.")
        # Create an empty in-memory state factory
        state_factory = LinkStateFactory(
            FLWR_IN_MEMORY_DB_NAME, NoOpFederationManager(), ObjectStoreFactory()
        )
        log(INFO, "Created new %s.", state_factory.__class__.__name__)

    if num_supernodes:
        # Register SuperNodes
        nodes_mapping = _register_nodes(
            num_nodes=num_supernodes,
            state_factory=state_factory,
            random_seed=random_seed,
        )

    # Construct mapping of DeprecatedRunInfoStore
    node_info_stores = _register_node_info_stores(
        nodes_mapping=nodes_mapping, run=run, app_dir=app_dir if is_app else None
    )

    # Load backend config
    backend_config = json.loads(backend_config_json_stream)

    def backend_fn() -> Backend:
        """Instantiate a Backend."""
        from .backend.raybackend import RayBackend  # pylint: disable=C0415

        return RayBackend(backend_config)

    # Load ClientApp if needed
    def _load() -> ClientApp:

        if client_app:
            return client_app
        if client_app_attr:
            return get_load_client_app_fn(
                default_app_ref=client_app_attr,
                app_path=app_dir,
                multi_app=False,
            )(run.fab_id, run.fab_version, run.fab_hash)

        raise ValueError("Either `client_app_attr` or `client_app` must be provided")

    return _load, backend_fn, nodes_mapping, state_factory, node_info_stores


def _import_client_app(
    app_fn: Callable[[], ClientApp],
    client_app_attr: str | None,
    f_stop: threading.Event,
) -> Callable[[], ClientApp]:
    """Import the ClientApp once.

    A reference load is cached so later calls do not import the module again.
    """
    try:
        loaded = app_fn()
    except LoadClientAppError as loadapp_ex:
        f_stop_delay = 10
        log(
            ERROR,
            "LoadClientAppError exception encountered. Terminating simulation in %is",
            f_stop_delay,
        )
        time.sleep(f_stop_delay)
        f_stop.set()  # set termination event
        raise loadapp_ex

    if not client_app_attr:
        return app_fn

    def _cached() -> ClientApp:
        return loaded

    return _cached


# pylint: disable=too-many-arguments,unused-argument,too-many-positional-arguments
def start_vce(
    backend_name: str,
    backend_config_json_stream: str,
    app_dir: str,
    is_app: bool,
    f_stop: threading.Event,
    run: Run,
    metrics: VceMetrics,
    client_app: ClientApp | None = None,
    client_app_attr: str | None = None,
    num_supernodes: int | None = None,
    state_factory: LinkStateFactory | None = None,
    existing_nodes_mapping: NodeToPartitionMapping | None = None,
    random_seed: int | None = None,
) -> None:
    """Start Fleet API with the Simulation Runtime."""
    app_fn, backend_fn, nodes_mapping, state_factory, node_info_stores = (
        _vce_components(
            backend_config_json_stream=backend_config_json_stream,
            app_dir=app_dir,
            is_app=is_app,
            run=run,
            client_app=client_app,
            client_app_attr=client_app_attr,
            num_supernodes=num_supernodes,
            state_factory=state_factory,
            existing_nodes_mapping=existing_nodes_mapping,
            random_seed=random_seed,
        )
    )
    app_fn = _import_client_app(app_fn, client_app_attr, f_stop)
    run_api(
        app_fn,
        backend_fn,
        nodes_mapping,
        state_factory,
        node_info_stores,
        f_stop,
        metrics,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def prepare_seeded_vce(
    backend_config_json_stream: str,
    app_dir: str,
    is_app: bool,
    f_stop: threading.Event,
    run: Run,
    random_seed: int,
    client_app: ClientApp | None = None,
    client_app_attr: str | None = None,
    num_supernodes: int | None = None,
    state_factory: LinkStateFactory | None = None,
) -> PreparedVce:
    """Register every node, import the ClientApp, and build the backend.

    Seeded simulations do this on the main thread before the ServerApp starts.
    The first sample then sees the full node set, and driver startup does not
    draw from the server RNGs.
    """
    app_fn, backend_fn, nodes_mapping, state_factory, node_info_stores = (
        _vce_components(
            backend_config_json_stream=backend_config_json_stream,
            app_dir=app_dir,
            is_app=is_app,
            run=run,
            client_app=client_app,
            client_app_attr=client_app_attr,
            num_supernodes=num_supernodes,
            state_factory=state_factory,
            existing_nodes_mapping=None,
            random_seed=random_seed,
        )
    )
    app_fn = _import_client_app(app_fn, client_app_attr, f_stop)
    backend = backend_fn()
    try:
        backend.build(app_fn)
    except Exception:
        # run_api's cleanup never sees a backend that failed during preparation.
        backend.terminate()
        raise
    return PreparedVce(
        app_fn=app_fn,
        nodes_mapping=nodes_mapping,
        state_factory=state_factory,
        node_info_stores=node_info_stores,
        backend=backend,
    )


def run_prepared_vce(
    prepared: PreparedVce,
    f_stop: threading.Event,
    metrics: VceMetrics,
) -> None:
    """Run a simulation runtime that :func:`prepare_seeded_vce` already built."""
    run_api(
        prepared.app_fn,
        lambda: prepared.backend,
        prepared.nodes_mapping,
        prepared.state_factory,
        prepared.node_info_stores,
        f_stop,
        metrics,
        backend=prepared.backend,
    )
