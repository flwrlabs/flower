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
"""Flower ClientProxy implementation using Grid."""


from flwr import common
from flwr.app.message_type import MessageType
from flwr.common import Message, MessageTypeLegacy, RecordDict
from flwr.common import recorddict_compat as compat
from flwr.server.client_proxy import ClientProxy

from ..grid.grid import Grid


class GridClientProxy(ClientProxy):
    """Flower client proxy which delegates work using Grid."""

    def __init__(self, node_id: int, grid: Grid, run_id: int):
        super().__init__(str(node_id))
        self.node_id = node_id
        self.grid = grid
        self.run_id = run_id

    def get_properties(
        self,
        ins: common.GetPropertiesIns,
        timeout: float | None,
        group_id: int | None,
    ) -> common.GetPropertiesRes:
        """Return client's properties."""
        # Ins to RecordDict
        out_recorddict = compat.getpropertiesins_to_recorddict(ins)
        # Fetch response
        in_recorddict = self._send_receive_recorddict(
            out_recorddict, MessageTypeLegacy.GET_PROPERTIES, timeout, group_id
        )
        # RecordDict to Res
        return compat.recorddict_to_getpropertiesres(in_recorddict)

    def get_parameters(
        self,
        ins: common.GetParametersIns,
        timeout: float | None,
        group_id: int | None,
    ) -> common.GetParametersRes:
        """Return the current local model parameters."""
        # Ins to RecordDict
        out_recorddict = compat.getparametersins_to_recorddict(ins)
        # Fetch response
        in_recorddict = self._send_receive_recorddict(
            out_recorddict, MessageTypeLegacy.GET_PARAMETERS, timeout, group_id
        )
        # RecordDict to Res
        return compat.recorddict_to_getparametersres(in_recorddict, False)

    def fit(
        self, ins: common.FitIns, timeout: float | None, group_id: int | None
    ) -> common.FitRes:
        """Train model parameters on the locally held dataset."""
        # Ins to RecordDict
        out_recorddict = compat.fitins_to_recorddict(ins, keep_input=True)
        # Fetch response
        in_recorddict = self._send_receive_recorddict(
            out_recorddict, MessageType.TRAIN, timeout, group_id
        )
        # RecordDict to Res
        return compat.recorddict_to_fitres(in_recorddict, keep_input=False)

    def evaluate(
        self, ins: common.EvaluateIns, timeout: float | None, group_id: int | None
    ) -> common.EvaluateRes:
        """Evaluate model parameters on the locally held dataset."""
        # Ins to RecordDict
        out_recorddict = compat.evaluateins_to_recorddict(ins, keep_input=True)
        # Fetch response
        in_recorddict = self._send_receive_recorddict(
            out_recorddict, MessageType.EVALUATE, timeout, group_id
        )
        # RecordDict to Res
        return compat.recorddict_to_evaluateres(in_recorddict)

    def reconnect(
        self,
        ins: common.ReconnectIns,
        timeout: float | None,
        group_id: int | None,
    ) -> common.DisconnectRes:
        """Disconnect and (optionally) reconnect later."""
        return common.DisconnectRes(reason="")  # Nothing to do here (yet)

    def _send_receive_recorddict(
        self,
        recorddict: RecordDict,
        message_type: str,
        timeout: float | None,
        group_id: int | None,
    ) -> RecordDict:

        # Create message
        message = Message(
            content=recorddict,
            message_type=message_type,
            dst_node_id=self.node_id,
            group_id=str(group_id) if group_id else "",
            ttl=timeout,
        )

        # Send message and wait for reply
        messages = list(self.grid.send_and_receive(messages=[message]))

        # A single reply is expected
        if len(messages) != 1:
            raise ValueError(f"Expected one Message but got: {len(messages)}")

        # Only messages without errors can be handled beyond these point
        msg: Message = messages[0]
        if msg.has_error():
            raise ValueError(
                f"Message contains an Error (reason: {msg.error.reason}). "
                "It originated during client-side execution of a message."
            )
        return msg.content
