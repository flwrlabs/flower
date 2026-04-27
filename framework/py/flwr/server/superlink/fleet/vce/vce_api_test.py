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
"""Test Fleet Simulation Engine API."""


import threading
from itertools import cycle
from json import JSONDecodeError
from math import pi
from pathlib import Path
from time import sleep
from unittest import TestCase

from flwr.client import Client, NumPyClient
from flwr.clientapp import ClientApp
from flwr.clientapp.client_app import LoadClientAppError
from flwr.common import (
    DEFAULT_TTL,
    Config,
    ConfigRecord,
    Context,
    GetPropertiesIns,
    MessageTypeLegacy,
    Metadata,
    RecordDict,
    Scalar,
    now,
)
from flwr.common.constant import Status
from flwr.common.message import make_message
from flwr.common.recorddict_compat import getpropertiesins_to_recorddict
from flwr.common.typing import Run, RunStatus
from flwr.server.superlink.fleet.vce.vce_api import (
    NodeToPartitionMapping,
    _register_nodes,
    start_vce,
)
from flwr.server.superlink.linkstate import InMemoryLinkState, LinkStateFactory
from flwr.server.superlink.linkstate.in_memory_linkstate import RunRecord
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink.federation import NoOpFederationManager


class DummyClient(NumPyClient):
    """A dummy NumPyClient for tests."""

    def __init__(self, state: RecordDict) -> None:
        self.client_state = state

    def get_properties(self, config: Config) -> dict[str, Scalar]:
        """Return properties by doing a simple calculation."""
        result = float(config["factor"]) * pi

        # store something in context
        self.client_state.config_records["result"] = ConfigRecord({"result": result})

        return {"result": result}


def get_dummy_client(context: Context) -> Client:  # pylint: disable=unused-argument
    """Return a DummyClient converted to Client type."""
    return DummyClient(state=context.state).to_client()


dummy_client_app = ClientApp(
    client_fn=get_dummy_client,
)


def terminate_simulation(f_stop: threading.Event, sleep_duration: int) -> None:
    """Set event to terminate Simulation Engine after `sleep_duration` seconds."""
    sleep(sleep_duration)
    f_stop.set()


def init_state_factory_nodes_mapping(
    num_nodes: int,
    num_messages: int,
) -> tuple[LinkStateFactory, NodeToPartitionMapping, dict[str, float]]:
    """Instatiate StateFactory, register nodes and pre-insert messages in the state."""
    # Register a state and a run_id in it
    run_id = 1234
    state_factory = LinkStateFactory(
        FLWR_IN_MEMORY_DB_NAME, NoOpFederationManager(), ObjectStoreFactory()
    )

    # Register a few nodes
    nodes_mapping = _register_nodes(num_nodes=num_nodes, state_factory=state_factory)

    expected_results = register_messages_into_state(
        state_factory=state_factory,
        nodes_mapping=nodes_mapping,
        run_id=run_id,
        num_messages=num_messages,
    )
    return state_factory, nodes_mapping, expected_results


# pylint: disable=too-many-locals
def register_messages_into_state(
    state_factory: LinkStateFactory,
    nodes_mapping: NodeToPartitionMapping,
    run_id: int,
    num_messages: int,
) -> dict[str, float]:
    """Register `num_messages` into the state factory."""
    state: InMemoryLinkState = state_factory.state()  # type: ignore
    state.run_ids[run_id] = RunRecord(
        Run(
            run_id=run_id,
            fab_id="Mock/mock",
            fab_version="v1.0.0",
            fab_hash="hash",
            override_config={},
            pending_at=now().isoformat(),
            starting_at="",
            running_at="",
            finished_at="",
            status=RunStatus(
                status=Status.PENDING,
                sub_status="",
                details="",
            ),
            flwr_aid="user123",
            federation="mock-fed",
            primary_task_id=None,
            bytes_sent=0,
            bytes_recv=0,
            clientapp_runtime=0.0,
        ),
    )
    # Artificially add Messages to state so they can be processed
    # by the Simulation Engine logic
    nodes_cycle = cycle(nodes_mapping.keys())  # we have more messages than supernodes
    message_ids: set[str] = set()  # so we can retrieve them later
    expected_results = {}
    for i in range(num_messages):
        dst_node_id = next(nodes_cycle)
        # Construct a Message
        mult_factor = 2024 + i
        getproperties_ins = GetPropertiesIns(config={"factor": mult_factor})
        recorddict = getpropertiesins_to_recorddict(getproperties_ins)
        message = make_message(
            content=recorddict,
            metadata=Metadata(
                run_id=run_id,
                message_id="",
                group_id="",
                src_node_id=0,
                dst_node_id=dst_node_id,  # indicate destination node
                reply_to_message_id="",
                created_at=now().timestamp(),
                ttl=DEFAULT_TTL,
                message_type=MessageTypeLegacy.GET_PROPERTIES,
            ),
        )

        # Insert in state
        message_id = state.store_message_res(message)
        if message_id:
            # Add message_id to set
            message_ids.add(message_id)
            # Store expected output for check later on
            expected_results[message_id] = mult_factor * pi

    return expected_results


def _autoresolve_app_dir(rel_client_app_dir: str = "backend") -> str:
    """Correctly resolve working directory for the app."""
    file_path = Path(__file__)
    app_dir = Path.cwd()
    rel_app_dir = file_path.relative_to(app_dir)

    # Susbtract lats element and append "backend/test" (wher the client module is.)
    return str(rel_app_dir.parent / rel_client_app_dir)


# pylint: disable=too-many-arguments,too-many-positional-arguments
def start_and_shutdown(
    backend: str = "ray",
    client_app_attr: str | None = None,
    app_dir: str = "",
    num_supernodes: int | None = None,
    state_factory: LinkStateFactory | None = None,
    nodes_mapping: NodeToPartitionMapping | None = None,
    duration: int = 0,
    backend_config: str = "{}",
) -> None:
    """Start Simulation Engine and terminate after specified number of seconds.

    Some tests need to be terminated by triggering externally an threading.Event. This
    is enabled when passing `duration`>0.
    """
    f_stop = threading.Event()

    if duration:

        # Setup thread that will set the f_stop event, triggering the termination of all
        # logic in the Simulation Engine. It will also terminate the Backend.
        termination_th = threading.Thread(
            target=terminate_simulation, args=(f_stop, duration)
        )
        termination_th.start()

    # Resolve working directory if not passed
    if not app_dir:
        app_dir = _autoresolve_app_dir()

    run = Run.create_empty(run_id=1234)

    start_vce(
        num_supernodes=num_supernodes,
        client_app=None if client_app_attr else dummy_client_app,
        client_app_attr=client_app_attr,
        backend_name=backend,
        backend_config_json_stream=backend_config,
        state_factory=state_factory,
        app_dir=app_dir,
        is_app=False,
        f_stop=f_stop,
        run=run,
        existing_nodes_mapping=nodes_mapping,
    )

    if duration:
        termination_th.join()


class TestFleetSimulationEngineRayBackend(TestCase):
    """A basic class that enables testing functionalities."""

    def test_erroneous_no_supernodes_client_mapping(self) -> None:
        """Test with unset arguments."""
        with self.assertRaises(ValueError):
            start_and_shutdown(duration=2)

    def test_erroneous_client_app_attr(self) -> None:
        """Tests attempt to load a ClientApp that can't be found."""
        num_messages = 7
        num_nodes = 59

        state_factory, nodes_mapping, _ = init_state_factory_nodes_mapping(
            num_nodes=num_nodes, num_messages=num_messages
        )
        with self.assertRaises(LoadClientAppError):
            start_and_shutdown(
                client_app_attr="totally_fictitious_app:client",
                state_factory=state_factory,
                nodes_mapping=nodes_mapping,
            )

    def test_erroneous_backend_config(self) -> None:
        """Backend Config should be a JSON stream."""
        with self.assertRaises(JSONDecodeError):
            start_and_shutdown(num_supernodes=50, backend_config="not a proper config")

    def test_with_nonexistent_backend(self) -> None:
        """Test specifying a backend that does not exist."""
        with self.assertRaises(KeyError):
            start_and_shutdown(num_supernodes=50, backend="this-backend-does-not-exist")

    def test_erroneous_arguments_num_supernodes_and_existing_mapping(self) -> None:
        """Test ValueError if a node mapping is passed but also num_supernodes.

        Passing `num_supernodes` does nothing since we assume that if a node mapping
        is supplied, nodes have been registered externally already. Therefore passing
        `num_supernodes` might give the impression that that many nodes will be
        registered. We don't do that since a mapping already exists.
        """
        with self.assertRaises(ValueError):
            start_and_shutdown(num_supernodes=50, nodes_mapping={0: 1})

    def test_erroneous_arguments_existing_mapping_but_no_state_factory(self) -> None:
        """Test ValueError if a node mapping is passed but no state.

        Passing a node mapping indicates that (externally) nodes have registered with a
        state factory. Therefore, that state factory should be passed too.
        """
        with self.assertRaises(ValueError):
            start_and_shutdown(nodes_mapping={0: 1})

    def test_start_and_shutdown(self) -> None:
        """Start Simulation Engine Fleet and terminate it."""
        start_and_shutdown(num_supernodes=50, duration=10)

    # pylint: disable=too-many-locals
    def test_start_and_shutdown_with_message_in_state(self) -> None:
        """Run Simulation Engine with some Message in State.

        This test creates a few nodes and submits a few messages that need to be
        executed by the Backend. In order for that to happen the asyncio
        producer/consumer logic must function. This also severs to evaluate a valid
        ClientApp.
        """
        num_messages = 229
        num_nodes = 59

        state_factory, nodes_mapping, expected_results = (
            init_state_factory_nodes_mapping(
                num_nodes=num_nodes,
                num_messages=num_messages,
            )
        )

        # Run
        start_and_shutdown(
            state_factory=state_factory, nodes_mapping=nodes_mapping, duration=10
        )

        # Get all Message replies
        state = state_factory.state()
        message_ids = set(expected_results.keys())
        message_res_list = state.get_message_res(message_ids=message_ids)

        # Check results by first converting to Message
        for message_res in message_res_list:

            # Verify message content is as expected
            content = message_res.content
            assert (
                content.config_records["getpropertiesres.properties"]["result"]
                == expected_results[message_res.metadata.reply_to_message_id]
            )
