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
