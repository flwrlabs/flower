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
"""Flower simulation tests."""


from math import pi
from random import shuffle

import ray

from flwr.app import ConfigRecord, Context, Message, RecordDict
from flwr.client import Client, NumPyClient
from flwr.clientapp import ClientApp
from flwr.common import Config, MessageTypeLegacy, Scalar
from flwr.common.constant import NUM_PARTITIONS_KEY, PARTITION_ID_KEY
from flwr.compat.client.run_info_store import DeprecatedRunInfoStore
from flwr.compat.common.recorddict_compat import (
    getpropertiesins_to_recorddict,
    recorddict_to_getpropertiesres,
)
from flwr.compat.common.recorddict_compat_test import _get_valid_getpropertiesins
from flwr.simulation.legacy_app import (
    NodeToPartitionMapping,
    _create_node_id_to_partition_mapping,
)
from flwr.simulation.ray_transport.ray_actor import (
    ClientAppActor,
    VirtualClientEngineActor,
    VirtualClientEngineActorPool,
)
from flwr.simulation.ray_transport.ray_client_proxy import RayActorClientProxy


class DummyClient(NumPyClient):
    """A dummy NumPyClient for tests."""

    def __init__(self, node_id: int, state: RecordDict) -> None:
        self.node_id = node_id
        self.client_state = state

    def get_properties(self, config: Config) -> dict[str, Scalar]:
        """Return properties by doing a simple calculation."""
        result = self.node_id * pi
        # store something in context
        self.client_state.config_records["result"] = ConfigRecord(
            {"result": str(result)}
        )
        return {"result": result}


def get_dummy_client(context: Context) -> Client:
    """Return a DummyClient converted to Client type."""
    return DummyClient(context.node_id, state=context.state).to_client()


def prep(
    actor_type: type[VirtualClientEngineActor] = ClientAppActor,
) -> tuple[
    list[RayActorClientProxy], VirtualClientEngineActorPool, NodeToPartitionMapping
]:  # pragma: no cover
    """Prepare ClientProxies and pool for tests."""
    client_resources = {"num_cpus": 1, "num_gpus": 0.0}

    def create_actor_fn() -> type[VirtualClientEngineActor]:
        return actor_type.options(**client_resources).remote()  # type: ignore

    # Create actor pool
    ray.init(include_dashboard=False)
    pool = VirtualClientEngineActorPool(
        create_actor_fn=create_actor_fn,
        client_resources=client_resources,
    )

    # Create 373 client proxies
    num_proxies = 373  # a prime number
    mapping = _create_node_id_to_partition_mapping(num_proxies)
    proxies = [
        RayActorClientProxy(
            client_fn=get_dummy_client,
            node_id=node_id,
            partition_id=partition_id,
            num_partitions=num_proxies,
            actor_pool=pool,
        )
        for node_id, partition_id in mapping.items()
    ]

    return proxies, pool, mapping


def test_cid_consistency_one_at_a_time() -> None:
    """Test that ClientProxies get the result of client job they submit.

    Submit one job and waits for completion. Then submits the next and so on
    """
    proxies, _, _ = prep()

    getproperties_ins = _get_valid_getpropertiesins()
    recorddict = getpropertiesins_to_recorddict(getproperties_ins)

    # submit jobs one at a time
    for prox in proxies:
        message = prox._wrap_recorddict_in_message(  # pylint: disable=protected-access
            recorddict,
            MessageTypeLegacy.GET_PROPERTIES,
            timeout=None,
            group_id=0,
        )
        message_out = prox._submit_job(  # pylint: disable=protected-access
            message=message, timeout=None
        )

        res = recorddict_to_getpropertiesres(message_out.content)

        assert int(prox.node_id) * pi == res.properties["result"]

    ray.shutdown()


def test_cid_consistency_all_submit_first_run_consistency() -> None:
    """Test that ClientProxies get the result of client job they submit.

    All jobs are submitted at the same time. Then fetched one at a time. This also tests
    DeprecatedRunInfoStore (at each Proxy) and RunState basic functionality.
    """
    proxies, _, _ = prep()
    run_id = 0

    getproperties_ins = _get_valid_getpropertiesins()
    recorddict = getpropertiesins_to_recorddict(getproperties_ins)

    # submit all jobs (collect later)
    shuffle(proxies)
    for prox in proxies:
        # Register state
        prox.proxy_state.register_context(run_id=run_id)
        # Retrieve state
        state = prox.proxy_state.retrieve_context(run_id=run_id)

        message = prox._wrap_recorddict_in_message(  # pylint: disable=protected-access
            recorddict,
            message_type=MessageTypeLegacy.GET_PROPERTIES,
            timeout=None,
            group_id=0,
        )
        prox.actor_pool.submit_client_job(
            lambda a, a_fn, mssg, cid, state: a.run.remote(a_fn, mssg, cid, state),
            (prox.app_fn, message, str(prox.node_id), state),
        )

    # fetch results one at a time
    shuffle(proxies)
    for prox in proxies:
        message_out, updated_context = prox.actor_pool.get_client_result(
            str(prox.node_id), timeout=None
        )
        prox.proxy_state.update_context(run_id, context=updated_context)
        res = recorddict_to_getpropertiesres(message_out.content)

        assert prox.node_id * pi == res.properties["result"]
        assert (
            str(prox.node_id * pi)
            == prox.proxy_state.retrieve_context(run_id).state.config_records["result"][
                "result"
            ]
        )
    ray.shutdown()


def test_cid_consistency_without_proxies() -> None:
    """Test cid consistency of jobs submitted/retrieved to/from pool w/o ClientProxy."""
    _, pool, mapping = prep()
    node_ids = list(mapping.keys())

    # register DeprecatedRunInfoStores
    node_info_stores: dict[int, DeprecatedRunInfoStore] = {}
    for node_id, partition_id in mapping.items():
        node_info_stores[node_id] = DeprecatedRunInfoStore(
            node_id=node_id,
            node_config={
                PARTITION_ID_KEY: str(partition_id),
                NUM_PARTITIONS_KEY: str(len(node_ids)),
            },
        )

    getproperties_ins = _get_valid_getpropertiesins()
    recorddict = getpropertiesins_to_recorddict(getproperties_ins)

    def _load_app() -> ClientApp:
        return ClientApp(client_fn=get_dummy_client)

    # submit all jobs (collect later)
    shuffle(node_ids)
    run_id = 0
    for node_id in node_ids:
        message = Message(
            content=recorddict,
            dst_node_id=node_id,
            message_type=MessageTypeLegacy.GET_PROPERTIES,
            group_id=str(0),
        )
        message.metadata.__dict__["_run_id"] = run_id
        # register and retrieve context
        node_info_stores[node_id].register_context(run_id=run_id)
        context = node_info_stores[node_id].retrieve_context(run_id=run_id)
        partition_id_str = str(context.node_config[PARTITION_ID_KEY])
        pool.submit_client_job(
            lambda a, c_fn, j_fn, nid_, state: a.run.remote(c_fn, j_fn, nid_, state),
            (_load_app, message, partition_id_str, context),
        )

    # fetch results one at a time
    shuffle(node_ids)
    for node_id in node_ids:
        partition_id_str = str(mapping[node_id])
        message_out, _ = pool.get_client_result(partition_id_str, timeout=None)
        res = recorddict_to_getpropertiesres(message_out.content)
        assert node_id * pi == res.properties["result"]

    ray.shutdown()
