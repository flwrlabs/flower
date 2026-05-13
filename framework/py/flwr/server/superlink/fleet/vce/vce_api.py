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
"""Fleet Simulation Engine API."""


import json
import secrets
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from logging import DEBUG, ERROR, INFO, WARN
from pathlib import Path
from queue import Empty, Queue
from uuid import uuid4

from flwr.app.error import Error
from flwr.client.run_info_store import DeprecatedRunInfoStore
from flwr.clientapp.client_app import ClientApp, ClientAppException, LoadClientAppError
from flwr.clientapp.utils import get_load_client_app_fn
from flwr.common import Message
from flwr.common.constant import (
    HEARTBEAT_INTERVAL_INF,
    NOOP_ACCOUNT_NAME,
    NOOP_FLWR_AID,
    NUM_PARTITIONS_KEY,
    PARTITION_ID_KEY,
    ErrorCode,
)
from flwr.common.logger import log
from flwr.common.typing import Run
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink.federation import NoOpFederationManager

from .backend import Backend, error_messages_backends, supported_backends

NodeToPartitionMapping = dict[int, int]


def _register_nodes(
    num_nodes: int, state_factory: LinkStateFactory
) -> NodeToPartitionMapping:
    """Register nodes with the StateFactory and create node-id:partition-id mapping."""
    nodes_mapping: NodeToPartitionMapping = {}
    state = state_factory.state()
    for i in range(num_nodes):
        node_id = state.create_node(
            # No node authentication in simulation;
            # use NOOP_FLWR_AID as owner_aid and
            # use random bytes as public key
            NOOP_FLWR_AID,
            NOOP_ACCOUNT_NAME,
            secrets.token_bytes(32),
            heartbeat_interval=HEARTBEAT_INTERVAL_INF,
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


# pylint: disable=too-many-arguments,too-many-locals
def worker(
    messageins_queue: Queue[Message],
    messageres_queue: Queue[Message],
    node_info_store: dict[int, DeprecatedRunInfoStore],
    backend: Backend,
    f_stop: threading.Event,
) -> None:
    """Process messages from the queue, execute them, update context, and enqueue
    replies."""
    while not f_stop.is_set():
        out_mssg = None
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
            message_ins_list = state.get_message_ins(node_id=node_id, limit=1)
            for msg in message_ins_list:
                queue.put(msg)
        f_stop.wait(0.1)


def put_message_into_state(
    state: LinkState, queue: Queue[Message], f_stop: threading.Event
) -> None:
    """Store reply Messages into the LinkState from the queue."""
    while not f_stop.is_set():
        try:
            message_reply = queue.get(timeout=1.0)
            state.store_message_res(message_reply)
        except Empty:
            # queue is empty when timeout was triggered
            pass


# pylint: disable=too-many-positional-arguments
def run_api(
    app_fn: Callable[[], ClientApp],
    backend_fn: Callable[[], Backend],
    nodes_mapping: NodeToPartitionMapping,
    state_factory: LinkStateFactory,
    node_info_stores: dict[int, DeprecatedRunInfoStore],
    f_stop: threading.Event,
) -> None:
    """Run the VCE."""
    messageins_queue: Queue[Message] = Queue()
    messageres_queue: Queue[Message] = Queue()

    backend = None
    try:

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
                )
                for _ in range(backend.num_workers)
            ]

        extractor_th.join()
        injector_th.join()

    except Exception as ex:

        log(ERROR, "An exception occured!! %s", ex)
        log(ERROR, traceback.format_exc())
        log(WARN, "Stopping Simulation Engine.")

        # Raise exception
        raise RuntimeError("Simulation Engine crashed.") from ex

    finally:
        # Manually trigger stopping event
        f_stop.set()

        # Terminate backend
        if backend is not None:
            backend.terminate()


# pylint: disable=too-many-arguments,unused-argument,too-many-locals,too-many-branches
# pylint: disable=too-many-statements,too-many-positional-arguments
def start_vce(
    backend_name: str,
    backend_config_json_stream: str,
    app_dir: str,
    is_app: bool,
    f_stop: threading.Event,
    run: Run,
    client_app: ClientApp | None = None,
    client_app_attr: str | None = None,
    num_supernodes: int | None = None,
    state_factory: LinkStateFactory | None = None,
    existing_nodes_mapping: NodeToPartitionMapping | None = None,
) -> None:
    """Start Fleet API with the Simulation Engine."""
    nodes_mapping = {}

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
            num_nodes=num_supernodes, state_factory=state_factory
        )

    # Construct mapping of DeprecatedRunInfoStore
    node_info_stores = _register_node_info_stores(
        nodes_mapping=nodes_mapping, run=run, app_dir=app_dir if is_app else None
    )

    # Load backend config
    log(DEBUG, "Supported backends: %s", list(supported_backends.keys()))
    backend_config = json.loads(backend_config_json_stream)

    try:
        backend_type = supported_backends[backend_name]
    except KeyError as ex:
        log(
            ERROR,
            "Backend `%s`, is not supported. Use any of %s or add support "
            "for a new backend.",
            backend_name,
            list(supported_backends.keys()),
        )
        if backend_name in error_messages_backends:
            log(ERROR, error_messages_backends[backend_name])

        raise ex

    def backend_fn() -> Backend:
        """Instantiate a Backend."""
        return backend_type(backend_config)

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

    app_fn = _load

    try:
        # Test if ClientApp can be loaded
        client_app = app_fn()

        # Cache `ClientApp`
        if client_app_attr:
            # Now wrap the loaded ClientApp in a dummy function
            # this prevent unnecesary low-level loading of ClientApp
            def _load_client_app() -> ClientApp:
                return client_app

            app_fn = _load_client_app

        # Run main simulation loop
        run_api(
            app_fn,
            backend_fn,
            nodes_mapping,
            state_factory,
            node_info_stores,
            f_stop,
        )
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
    except Exception as ex:
        raise ex
