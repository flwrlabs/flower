# Copyright 2026 Inria (cyrille kenfack & davide frey). All Rights Reserved.
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
from dataclasses import dataclass

from flwr.common.message import Message
from flwr.common.record.recorddict import RecordDict

from .typing import Action


@dataclass(frozen=True)
class AggregateRequest:
    """Message exchanged between decentralized nodes for aggregation.

    Attributes
    ----------
    action : Action
        Action to be performed by decentralized nodes.
    source_node_id : str
        ID of the source node sending the message.
    destination_node_id : str
        ID of the destination node receiving the message.
    round_number : int
        Current round number of the decentralized training process.
    parameters : NDArrays
        Model parameters or updates being exchanged between nodes.
    """

    action: Action
    source_node_id: str
    round_number: int
    msg: Message | None = None

    def to_kwargs(self) -> dict:
        """Convert message to kwargs for easier handling."""
        return {
            "action": self.action,
            "source_node_id": self.source_node_id,
            "round_number": self.round_number,
            "msg": self.msg,
        }
