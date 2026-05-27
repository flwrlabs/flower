from enum import Enum

DEFAULT_IP_ADDRESS = "0.0.0.0"
DEFAULT_PORT = 0

class Mode(Enum):
    """Mode to be performed by decentralized nodes in the communication."""

    PUSHPULL = "PUSHPULL"
    PUSH = "PUSH"

class Action(Enum):
    """Action to be performed by decentralized nodes."""

    PUSHPULL = "PUSHPULL"
    PUSH = "PUSH"
    CANCEL = "CANCEL"