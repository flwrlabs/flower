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
"""Ray-based Flower ClientProxy implementation."""


import traceback
from logging import ERROR

from flwr import common
from flwr.app.message_type import MessageType
from flwr.client import ClientFnExt
from flwr.client.run_info_store import DeprecatedRunInfoStore
from flwr.clientapp.client_app import ClientApp
from flwr.common import DEFAULT_TTL, Message, Metadata, RecordDict, now
from flwr.common.constant import NUM_PARTITIONS_KEY, PARTITION_ID_KEY, MessageTypeLegacy
from flwr.common.logger import log
from flwr.common.message import make_message
from flwr.common.recorddict_compat import (
    evaluateins_to_recorddict,
    fitins_to_recorddict,
    getparametersins_to_recorddict,
    getpropertiesins_to_recorddict,
    recorddict_to_evaluateres,
    recorddict_to_fitres,
    recorddict_to_getparametersres,
    recorddict_to_getpropertiesres,
)
from flwr.server.client_proxy import ClientProxy
from flwr.simulation.ray_transport.ray_actor import VirtualClientEngineActorPool


class RayActorClientProxy(ClientProxy):
    """Flower client proxy which delegates work using Ray."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        client_fn: ClientFnExt,
        node_id: int,
        partition_id: int,
        num_partitions: int,
        actor_pool: VirtualClientEngineActorPool,
    ):
        super().__init__(cid=str(node_id))
        self.node_id = node_id
        self.partition_id = partition_id

        def _load_app() -> ClientApp:
            return ClientApp(client_fn=client_fn)

        self.app_fn = _load_app
        self.actor_pool = actor_pool
        self.proxy_state = DeprecatedRunInfoStore(
            node_id=node_id,
            node_config={
                PARTITION_ID_KEY: str(partition_id),
                NUM_PARTITIONS_KEY: str(num_partitions),
            },
        )

    def _submit_job(self, message: Message, timeout: float | None) -> Message:
        """Sumbit a message to the ActorPool."""
        run_id = message.metadata.run_id

        # Register state
        self.proxy_state.register_context(run_id=run_id)

        # Retrieve context
        context = self.proxy_state.retrieve_context(run_id=run_id)
        partition_id_str = str(context.node_config[PARTITION_ID_KEY])

        try:
            self.actor_pool.submit_client_job(
                lambda a, a_fn, mssg, partition_id, context: a.run.remote(
                    a_fn, mssg, partition_id, context
                ),
                (self.app_fn, message, partition_id_str, context),
            )
            out_mssg, updated_context = self.actor_pool.get_client_result(
                partition_id_str, timeout
            )

            # Update state
            self.proxy_state.update_context(run_id=run_id, context=updated_context)

        except Exception as ex:
            if self.actor_pool.num_actors == 0:
                # At this point we want to stop the simulation.
                # since no more client runs will be executed
                log(ERROR, "ActorPool is empty!!!")
            log(ERROR, traceback.format_exc())
            log(ERROR, ex)
            raise ex

        return out_mssg

    def _wrap_recorddict_in_message(
        self,
        recorddict: RecordDict,
        message_type: str,
        timeout: float | None,
        group_id: int | None,
    ) -> Message:
        """Wrap a RecordDict inside a Message."""
        return make_message(
            content=recorddict,
            metadata=Metadata(
                run_id=0,
                message_id="",
                group_id=str(group_id) if group_id is not None else "",
                src_node_id=0,
                dst_node_id=self.node_id,
                reply_to_message_id="",
                created_at=now().timestamp(),
                ttl=timeout if timeout else DEFAULT_TTL,
                message_type=message_type,
            ),
        )

    def get_properties(
        self,
        ins: common.GetPropertiesIns,
        timeout: float | None,
        group_id: int | None,
    ) -> common.GetPropertiesRes:
        """Return client's properties."""
        recorddict = getpropertiesins_to_recorddict(ins)
        message = self._wrap_recorddict_in_message(
            recorddict,
            message_type=MessageTypeLegacy.GET_PROPERTIES,
            timeout=timeout,
            group_id=group_id,
        )

        message_out = self._submit_job(message, timeout)

        return recorddict_to_getpropertiesres(message_out.content)

    def get_parameters(
        self,
        ins: common.GetParametersIns,
        timeout: float | None,
        group_id: int | None,
    ) -> common.GetParametersRes:
        """Return the current local model parameters."""
        recorddict = getparametersins_to_recorddict(ins)
        message = self._wrap_recorddict_in_message(
            recorddict,
            message_type=MessageTypeLegacy.GET_PARAMETERS,
            timeout=timeout,
            group_id=group_id,
        )

        message_out = self._submit_job(message, timeout)

        return recorddict_to_getparametersres(message_out.content, keep_input=False)

    def fit(
        self, ins: common.FitIns, timeout: float | None, group_id: int | None
    ) -> common.FitRes:
        """Train model parameters on the locally held dataset."""
        recorddict = fitins_to_recorddict(
            ins, keep_input=True
        )  # This must stay TRUE since ins are in-memory
        message = self._wrap_recorddict_in_message(
            recorddict,
            message_type=MessageType.TRAIN,
            timeout=timeout,
            group_id=group_id,
        )

        message_out = self._submit_job(message, timeout)

        return recorddict_to_fitres(message_out.content, keep_input=False)

    def evaluate(
        self, ins: common.EvaluateIns, timeout: float | None, group_id: int | None
    ) -> common.EvaluateRes:
        """Evaluate model parameters on the locally held dataset."""
        recorddict = evaluateins_to_recorddict(
            ins, keep_input=True
        )  # This must stay TRUE since ins are in-memory
        message = self._wrap_recorddict_in_message(
            recorddict,
            message_type=MessageType.EVALUATE,
            timeout=timeout,
            group_id=group_id,
        )

        message_out = self._submit_job(message, timeout)

        return recorddict_to_evaluateres(message_out.content)

    def reconnect(
        self,
        ins: common.ReconnectIns,
        timeout: float | None,
        group_id: int | None,
    ) -> common.DisconnectRes:
        """Disconnect and (optionally) reconnect later."""
        return common.DisconnectRes(reason="")  # Nothing to do here (yet)
