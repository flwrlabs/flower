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
"""Test for Ray backend for the Fleet API using the Simulation Runtime."""


from collections.abc import Callable
from math import pi
from unittest import TestCase

import ray

from flwr.app import DEFAULT_TTL, ConfigRecord, Context, Message, Metadata, RecordDict
from flwr.app.message import make_message
from flwr.client import Client, NumPyClient
from flwr.clientapp import ClientApp
from flwr.common import Config, GetPropertiesIns, MessageTypeLegacy, Scalar
from flwr.common.constant import PARTITION_ID_KEY
from flwr.compat.client.run_info_store import DeprecatedRunInfoStore
from flwr.compat.common.recorddict_compat import getpropertiesins_to_recorddict
from flwr.server.superlink.fleet.vce.backend.backend import BackendConfig
from flwr.server.superlink.fleet.vce.backend.raybackend import RayBackend
from flwr.simulation.ray_transport.ray_actor import pool_size_from_resources
from flwr.supercore.date import now


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


def _load_app() -> ClientApp:
    return ClientApp(client_fn=get_dummy_client)


def backend_build_process_and_termination(
    backend: RayBackend,
    app_fn: Callable[[], ClientApp],
    process_args: tuple[Message, Context] | None = None,
) -> tuple[Message, Context] | None:
    """Build, process job and terminate RayBackend."""
    backend.build(app_fn)
    to_return = None

    if process_args:
        to_return = backend.process_message(*process_args)

    backend.terminate()

    return to_return


def _create_message_and_context() -> tuple[Message, Context, float]:

    # Construct a Message
    mult_factor = 2024
    run_id = 0
    getproperties_ins = GetPropertiesIns(config={"factor": mult_factor})
    recorddict = getpropertiesins_to_recorddict(getproperties_ins)
    message = make_message(
        content=recorddict,
        metadata=Metadata(
            run_id=run_id,
            message_id="",
            group_id="",
            src_node_id=0,
            dst_node_id=0,
            reply_to_message_id="",
            created_at=now().timestamp(),
            ttl=DEFAULT_TTL,
            message_type=MessageTypeLegacy.GET_PROPERTIES,
        ),
    )

    # Construct DeprecatedRunInfoStore and retrieve context
    node_state = DeprecatedRunInfoStore(
        node_id=run_id, node_config={PARTITION_ID_KEY: str(0)}
    )
    node_state.register_context(run_id=run_id)
    context = node_state.retrieve_context(run_id=run_id)

    # Expected output
    expected_output = pi * mult_factor

    return message, context, expected_output


class TestRayBackend(TestCase):
    """A basic class that allows runnig multliple tests."""

    def doCleanups(self) -> None:
        """Ensure Ray has shutdown."""
        if ray.is_initialized():
            ray.shutdown()

    def test_backend_creation_and_termination(self) -> None:
        """Test creation of RayBackend and its termination."""
        backend = RayBackend(backend_config={})
        backend_build_process_and_termination(
            backend=backend, app_fn=_load_app, process_args=None
        )

    def test_backend_creation_submit_and_termination(
        self,
        client_app_loader: Callable[[], ClientApp] = _load_app,
    ) -> None:
        """Test submitting a message to a given ClientApp."""
        backend = RayBackend(backend_config={})

        message, context, expected_output = _create_message_and_context()

        res = backend_build_process_and_termination(
            backend=backend, app_fn=client_app_loader, process_args=(message, context)
        )

        if res is None:
            raise AssertionError("This shouldn't happen")

        out_mssg, updated_context = res

        # Verify message content is as expected
        content = out_mssg.content
        assert (
            content.config_records["getpropertiesres.properties"]["result"]
            == expected_output
        )
        # Verify context is correct
        obtained_result_in_context = updated_context.state.config_records["result"][
            "result"
        ]
        assert obtained_result_in_context == expected_output

    def test_backend_creation_submit_and_termination_existing_client_app(
        self,
    ) -> None:
        """Testing with ClientApp module that exist."""
        self.test_backend_creation_submit_and_termination(
            client_app_loader=_load_app,
        )

    def test_backend_creation_with_init_arguments(self) -> None:
        """Testing whether init args are properly parsed to Ray."""
        backend_config_4: BackendConfig = {
            "init_args": {"num_cpus": 4},
            "client_resources": {"num_cpus": 1, "num_gpus": 0},
        }

        backend_config_2: BackendConfig = {
            "init_args": {"num_cpus": 2},
            "client_resources": {"num_cpus": 1, "num_gpus": 0},
        }

        RayBackend(
            backend_config=backend_config_4,
        )
        nodes = ray.nodes()  # type: ignore[no-untyped-call]

        assert nodes[0]["Resources"]["CPU"] == backend_config_4["init_args"]["num_cpus"]

        ray.shutdown()

        RayBackend(
            backend_config=backend_config_2,
        )
        nodes = ray.nodes()  # type: ignore[no-untyped-call]

        assert nodes[0]["Resources"]["CPU"] == backend_config_2["init_args"]["num_cpus"]

    def test_case_with_no_cpu_resources_on_node(self) -> None:
        """Test mixed environment with zero and non-zero CPU nodes."""
        try:
            # Start Ray with head node (zero CPU)
            ray.init(num_cpus=0)

            # Mock ray.nodes() to return both head node and worker node
            original_nodes = ray.nodes

            head_node = ray.nodes()[0].copy()  # type: ignore[no-untyped-call]

            ray.nodes = lambda: [
                head_node,  # Head node initialized with no cpu
                {"Resources": {"CPU": 8}},  # Worker node with 8 CPUs
            ]

            try:
                # Test the pool size calculation
                client_resources: dict[str, int | float] = {
                    "num_cpus": 2,
                    "num_gpus": 0,
                }
                pool_size = pool_size_from_resources(client_resources)
                # Should calculate based on the worker node (8 CPUs)
                self.assertEqual(pool_size, 4)  # 8 / 2 CPUs required for each task
            finally:
                # Restore original functions
                ray.nodes = original_nodes

        finally:
            ray.shutdown()
